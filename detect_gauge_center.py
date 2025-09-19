from __future__ import annotations

import os
from glob import glob
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GaugeCalibration:
    angle_min: float
    angle_max: float
    value_min: float
    value_max: float
    clockwise: bool = True  # True si al aumentar el valor la aguja gira sentido horario


SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated")
RECTIFIED_DIR = os.path.join(OUTPUT_DIR, "rectified")
MASK_DIR = os.path.join(OUTPUT_DIR, "mask")

for path in (OUTPUT_DIR, ANNOTATED_DIR, RECTIFIED_DIR, MASK_DIR):
    os.makedirs(path, exist_ok=True)


# Completar estos valores con la geometría de cada gauge.
GAUGE_CALIBRATIONS: dict[str, GaugeCalibration] = {
    # "75.jpg": GaugeCalibration(angle_min=40, angle_max=320, value_min=0, value_max=160, clockwise=True),
}


def preprocess_for_contours(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return blurred


def find_ellipse(gray: np.ndarray) -> tuple[tuple | None, np.ndarray]:
    processed = preprocess_for_contours(gray)
    edges = cv2.Canny(processed, 40, 120)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, edges

    h, w = gray.shape
    img_area = float(h * w)
    best_contour = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.05 * img_area:
            continue
        if len(contour) < 5:
            continue
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter == 0.0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.5:
            continue
        if area > best_area:
            best_area = area
            best_contour = contour

    if best_contour is None:
        return None, edges

    ellipse = cv2.fitEllipse(best_contour)
    return ellipse, edges


def build_normalization_homography(center: tuple[float, float], axes: tuple[float, float], angle_deg: float) -> np.ndarray:
    cx, cy = center
    major_axis, minor_axis = axes
    if minor_axis <= 0:
        return np.eye(3, dtype=np.float32)

    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)

    translate_to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float32)
    rotate = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
    scale = np.array([[1, 0, 0], [0, major_axis / minor_axis, 0], [0, 0, 1]], dtype=np.float32)
    translate_back = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=np.float32)

    homography = translate_back @ scale @ rotate @ translate_to_origin
    return homography.astype(np.float32)


def rectify_gauge(img: np.ndarray, ellipse: tuple, crop_scale: float = 1.1, output_size: int = 512) -> tuple[np.ndarray, int]:
    (cx, cy), (major_axis, minor_axis), angle = ellipse
    h, w = img.shape[:2]
    homography = build_normalization_homography((cx, cy), (major_axis, minor_axis), angle)
    normalized = cv2.warpPerspective(img, homography, (w, h), flags=cv2.INTER_CUBIC)

    radius = int(max(major_axis, minor_axis) * 0.5 * crop_scale)
    radius = max(radius, 1)
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    x0 = cx_i - radius
    y0 = cy_i - radius
    x1 = cx_i + radius
    y1 = cy_i + radius

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        normalized = cv2.copyMakeBorder(
            normalized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT,
        )
        cx_i += pad_left
        cy_i += pad_top
        x0 += pad_left
        y0 += pad_top
        x1 += pad_left
        y1 += pad_top

    roi = normalized[y0:y1, x0:x1]
    if roi.size == 0:
        return None, radius
    roi = cv2.resize(roi, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    return roi, radius


def detect_pointer(rectified: np.ndarray) -> tuple[dict | None, np.ndarray]:
    h, w = rectified.shape[:2]
    center = (w // 2, h // 2)
    max_radius = min(center)

    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 110)

    mask = np.zeros_like(edges)
    outer = int(max_radius * 0.95)
    inner = int(max_radius * 0.25)
    cv2.circle(mask, center, outer, 255, -1)
    cv2.circle(mask, center, inner, 0, -1)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=int(outer * 0.6),
        maxLineGap=20,
    )

    if lines is None:
        return None, masked_edges

    best_line = None
    best_score = 0.0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        d1 = np.hypot(x1 - center[0], y1 - center[1])
        d2 = np.hypot(x2 - center[0], y2 - center[1])
        near_dist = min(d1, d2)
        far_dist = max(d1, d2)
        if near_dist > inner * 1.4:
            continue
        if far_dist < outer * 0.55:
            continue
        length = np.hypot(x1 - x2, y1 - y2)
        score = length + (far_dist - near_dist)
        if score > best_score:
            best_score = score
            best_line = (x1, y1, x2, y2)

    if best_line is None:
        return None, masked_edges

    x1, y1, x2, y2 = best_line
    d1 = np.hypot(x1 - center[0], y1 - center[1])
    d2 = np.hypot(x2 - center[0], y2 - center[1])
    if d1 < d2:
        near = (x1, y1)
        far = (x2, y2)
    else:
        near = (x2, y2)
        far = (x1, y1)

    dx = far[0] - center[0]
    dy = center[1] - far[1]
    angle_deg = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

    data = {
        "line": best_line,
        "near": near,
        "far": far,
        "angle_deg": angle_deg,
        "center": center,
    }
    return data, masked_edges


def unwrap_angle(angle: float, reference: float) -> float:
    wrapped = (angle - reference + 360.0) % 360.0
    return wrapped + reference


def angle_to_value(angle: float, calibration: GaugeCalibration) -> float:
    angle_min = calibration.angle_min
    angle_max = calibration.angle_max
    if calibration.clockwise:
        angle = (360.0 - angle) % 360.0
        angle_min = (360.0 - angle_min) % 360.0
        angle_max = (360.0 - angle_max) % 360.0

    start = unwrap_angle(angle_min, angle_min)
    end = unwrap_angle(angle_max, angle_min)
    current = unwrap_angle(angle, angle_min)

    span = end - start
    if span == 0.0:
        return calibration.value_min

    alpha = np.clip((current - start) / span, 0.0, 1.0)
    value = calibration.value_min + alpha * (calibration.value_max - calibration.value_min)
    return value


def annotate_outputs(original: np.ndarray, ellipse: tuple | None, pointer: dict | None, angle: float | None, value: float | None) -> np.ndarray:
    canvas = original.copy()
    if ellipse is not None:
        cv2.ellipse(canvas, tuple(ellipse), (0, 255, 0), 2)
        center = (int(round(ellipse[0][0])), int(round(ellipse[0][1])))
        cv2.drawMarker(canvas, center, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)

    if pointer is not None:
        x1, y1, x2, y2 = pointer["line"]
        cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 255), 2)

    label = []
    if angle is not None:
        label.append(f"Ángulo: {angle:0.1f}°")
    if value is not None:
        label.append(f"Valor: {value:0.2f}")
    if label:
        text = " | ".join(label)
        cv2.rectangle(canvas, (20, 20), (20 + 12 * len(text), 60), (0, 0, 0), -1)
        cv2.putText(canvas, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    return canvas


def process_image(image_path: str) -> None:
    image = cv2.imread(image_path)
    if image is None:
        print(f"No se pudo leer {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ellipse, edges = find_ellipse(gray)

    if ellipse is None:
        print(f"No se detectó elipse en {image_path}")
        base = os.path.basename(image_path)
        cv2.imwrite(os.path.join(ANNOTATED_DIR, base), image)
        cv2.imwrite(os.path.join(MASK_DIR, base), edges)
        return

    rectified, _ = rectify_gauge(image, ellipse)
    if rectified is None:
        print(f"{os.path.basename(image_path)}: no se pudo recortar el gauge")
        return

    pointer, needle_mask = detect_pointer(rectified)

    angle_deg = pointer["angle_deg"] if pointer else None
    value = None
    calib = GAUGE_CALIBRATIONS.get(os.path.basename(image_path))
    if angle_deg is not None and calib is not None:
        value = angle_to_value(angle_deg, calib)

    annotated_original = annotate_outputs(image, ellipse, pointer, angle_deg, value)
    annotated_rectified = rectified.copy()
    if pointer is not None:
        center = pointer["center"]
        cv2.circle(annotated_rectified, center, int(min(center) * 0.25), (128, 128, 128), 1)
        x1, y1, x2, y2 = pointer["line"]
        cv2.line(annotated_rectified, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.drawMarker(annotated_rectified, center, (0, 0, 255), cv2.MARKER_CROSS, 26, 2)

    base_name = os.path.basename(image_path)
    cv2.imwrite(os.path.join(ANNOTATED_DIR, base_name), annotated_original)
    cv2.imwrite(os.path.join(RECTIFIED_DIR, base_name), annotated_rectified)
    cv2.imwrite(os.path.join(MASK_DIR, base_name), needle_mask)

    if angle_deg is not None:
        message = f"{base_name}: ángulo {angle_deg:0.1f}°"
        if value is not None:
            message += f", valor {value:0.2f}"
        print(message)
    else:
        print(f"{base_name}: no se detectó aguja")


def main() -> None:
    image_paths = glob(os.path.join(SAMPLES_DIR, "*.*"))
    if not image_paths:
        print(f"No se encontraron imágenes en {SAMPLES_DIR}")
        return
    for image_path in image_paths:
        process_image(image_path)


if __name__ == "__main__":
    main()
