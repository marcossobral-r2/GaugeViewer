from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from glob import glob
from typing import Iterable

import cv2
import numpy as np


@dataclass
class GaugeCalibration:
    angle_min: float
    angle_max: float
    value_min: float
    value_max: float
    clockwise: bool = True  # True si al aumentar el valor la aguja gira sentido horario

    @classmethod
    def from_mapping(cls, data: dict) -> "GaugeCalibration":
        required = {"angle_min", "angle_max", "value_min", "value_max"}
        missing = required - data.keys()
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Faltan campos en la calibración: {missing_str}")
        return cls(
            angle_min=float(data["angle_min"]),
            angle_max=float(data["angle_max"]),
            value_min=float(data["value_min"]),
            value_max=float(data["value_max"]),
            clockwise=bool(data.get("clockwise", True)),
        )


PROJECT_ROOT = os.path.dirname(__file__)
DEFAULT_SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


@dataclass
class OutputPaths:
    base: str
    annotated: str
    rectified: str
    mask: str

    @classmethod
    def from_base(cls, base: str) -> "OutputPaths":
        annotated = os.path.join(base, "annotated")
        rectified = os.path.join(base, "rectified")
        mask = os.path.join(base, "mask")
        return cls(base=base, annotated=annotated, rectified=rectified, mask=mask)

    def ensure(self) -> None:
        for path in (self.base, self.annotated, self.rectified, self.mask):
            os.makedirs(path, exist_ok=True)


# Completar estos valores con la geometría de cada gauge.
GAUGE_CALIBRATIONS: dict[str, GaugeCalibration] = {
    # "75.jpg": GaugeCalibration(angle_min=40, angle_max=320, value_min=0, value_max=160, clockwise=True),
}


def load_calibrations_from_file(path: str) -> dict[str, GaugeCalibration]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("El archivo de calibración debe contener un objeto JSON de nivel superior")

    calibrations: dict[str, GaugeCalibration] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"El valor asociado a '{key}' debe ser un objeto JSON")
        calibrations[key] = GaugeCalibration.from_mapping(value)
    return calibrations


def resolve_image_inputs(inputs: Iterable[str] | None, default_dir: str) -> list[str]:
    if not inputs:
        pattern = os.path.join(default_dir, "*.*")
        return sorted(glob(pattern))

    resolved: list[str] = []
    for entry in inputs:
        entry = os.path.expanduser(entry)
        if os.path.isdir(entry):
            pattern = os.path.join(entry, "*.*")
            matches = sorted(glob(pattern))
            resolved.extend(matches)
            continue

        matches = sorted(glob(entry))
        if matches:
            resolved.extend(matches)
            continue

        resolved.append(entry)

    # Mantener el orden pero eliminar duplicados.
    unique: list[str] = []
    seen: set[str] = set()
    for path in resolved:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def preprocess_for_contours(gray: np.ndarray) -> np.ndarray:
    """Enhance contrast and remove noise before contour extraction."""

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Median blur is more robust to salt-and-pepper noise than Gaussian for dials
    # with engraved text or dust.
    median = cv2.medianBlur(enhanced, 5)
    bilateral = cv2.bilateralFilter(median, d=7, sigmaColor=50, sigmaSpace=50)
    return bilateral


def auto_canny(image: np.ndarray, sigma: float = 0.33, aperture_size: int = 3) -> np.ndarray:
    """Compute Canny edges with thresholds derived from the median intensity."""

    v = float(np.median(image))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    if lower >= upper:
        upper = min(255, lower + 1)
    edges = cv2.Canny(image, lower, upper, apertureSize=aperture_size)
    return edges


def detect_dial_circle(gray: np.ndarray) -> tuple[float, float, float] | None:
    """Locate the gauge ring by sweeping a compact bank of Hough parameters."""

    h, w = gray.shape
    preprocessed = preprocess_for_contours(gray)
    min_radius = max(10, int(min(h, w) * 0.18))
    max_radius = int(min(h, w) * 0.72)

    # Parameter bank ordered from strict to permissive; the first viable hit typically
    # comes from a high-threshold configuration, keeping runtime low while still
    # letting the loop relax thresholds for dim or noisy dials.
    parameter_bank = (
        (7, 1.2, 180, 110),
        (9, 1.2, 170, 95),
        (7, 1.05, 170, 85),
        (5, 1.0, 160, 75),
        (9, 1.35, 170, 65),
        (7, 1.0, 150, 60),
        (5, 1.2, 150, 52),
        (7, 1.35, 150, 45),
    )

    best_circle: tuple[float, float, float] | None = None
    best_score = -np.inf
    image_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    for blur_size, dp, param1, param2 in parameter_bank:
        blurred = cv2.GaussianBlur(preprocessed, (blur_size, blur_size), 0)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min(h, w) * 0.5,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            continue

        for x, y, r in circles[0]:
            if not (0.0 <= x < w and 0.0 <= y < h):
                continue
            radius = float(r)
            center = np.array([float(x), float(y)], dtype=np.float32)
            center_penalty = np.linalg.norm(center - image_center)
            score = radius * 1.4 - 0.75 * center_penalty + param2
            if score > best_score:
                best_score = score
                best_circle = (float(x), float(y), radius)

        # Si encontramos una circunferencia con radio grande y el centro bien
        # posicionado es muy probable que sea el dial, por lo que podemos
        # abandonar la búsqueda tempranamente.
        if best_circle is not None and best_score > 150:
            break

    return best_circle


def _refine_ellipse_from_circle(gray: np.ndarray, circle: tuple[float, float, float]) -> tuple[tuple | None, np.ndarray]:
    cx, cy, radius = circle
    processed = preprocess_for_contours(gray)
    edges = auto_canny(processed, sigma=0.2)

    mask = np.zeros_like(edges)
    center = (int(round(cx)), int(round(cy)))
    outer = max(1, int(radius * 1.35))
    inner = max(1, int(radius * 0.55))
    cv2.circle(mask, center, outer, 255, -1)
    cv2.circle(mask, center, inner, 0, -1)

    masked_edges = cv2.bitwise_and(edges, mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masked_edges = cv2.morphologyEx(masked_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    masked_edges = cv2.morphologyEx(masked_edges, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(masked_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, masked_edges

    best_ellipse: tuple | None = None
    best_score = -np.inf
    for contour in contours:
        if len(contour) < 5:
            continue
        area = cv2.contourArea(contour)
        if area < 50.0:
            continue

        ellipse = cv2.fitEllipse(contour)
        (ex, ey), (major_axis, minor_axis), angle = ellipse
        if minor_axis <= 0:
            continue

        ratio = max(major_axis, minor_axis) / max(1.0, min(major_axis, minor_axis))
        if ratio > 3.0:
            continue

        major_radius = 0.5 * max(major_axis, minor_axis)
        minor_radius = 0.5 * min(major_axis, minor_axis)
        if not (radius * 0.45 <= major_radius <= radius * 1.8):
            continue
        if not (radius * 0.3 <= minor_radius <= radius * 1.4):
            continue

        center_offset = np.hypot(ex - cx, ey - cy)
        avg_radius = 0.25 * (major_axis + minor_axis)
        radius_error = abs(avg_radius - radius)
        score = area - 10.0 * center_offset - 6.0 * radius_error
        if score > best_score:
            best_score = score
            best_ellipse = ellipse

    if best_ellipse is not None:
        return best_ellipse, masked_edges

    return ((cx, cy), (radius * 2.0, radius * 2.0), 0.0), masked_edges


def _contour_based_ellipse(gray: np.ndarray) -> tuple[tuple | None, np.ndarray]:
    processed = preprocess_for_contours(gray)
    edges = auto_canny(processed, sigma=0.2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, edges

    h, w = gray.shape
    img_area = float(h * w)
    best_contour = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.02 * img_area:
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
    (cx, cy), (major_axis, minor_axis), _ = ellipse
    if not (0 <= cx < w and 0 <= cy < h):
        return None, edges
    if minor_axis <= 0:
        return None, edges

    axis_ratio = max(major_axis, minor_axis) / max(1.0, min(major_axis, minor_axis))
    if axis_ratio > 2.2:
        # Cuando el ajuste es extremadamente oblongo suele tratarse de un contorno espurio.
        return None, edges

    ellipse_area = np.pi * (major_axis * 0.5) * (minor_axis * 0.5)
    if ellipse_area < 0.1 * img_area:
        return None, edges

    return ellipse, edges


def find_ellipse(gray: np.ndarray) -> tuple[tuple | None, np.ndarray]:
    circle = detect_dial_circle(gray)
    if circle is not None:
        ellipse, edges = _refine_ellipse_from_circle(gray, circle)
        if ellipse is not None:
            return ellipse, edges

    return _contour_based_ellipse(gray)


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


def _score_pointer_candidate(
    center: tuple[int, int],
    inner: float,
    outer: float,
    p1: tuple[float, float],
    p2: tuple[float, float],
) -> tuple[float, tuple[int, int], tuple[int, int], float] | None:
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.hypot(dx, dy))
    if length < max(outer * 0.28, 22.0):
        return None

    denom = length if length > 1e-6 else 1e-6
    line_distance = abs((y2 - y1) * center[0] - (x2 - x1) * center[1] + x2 * y1 - y2 * x1) / denom
    if line_distance > inner * 1.1:
        return None

    d1 = float(np.hypot(x1 - center[0], y1 - center[1]))
    d2 = float(np.hypot(x2 - center[0], y2 - center[1]))
    near_dist = min(d1, d2)
    far_dist = max(d1, d2)
    if near_dist > inner * 1.7:
        return None
    if far_dist < outer * 0.3:
        return None

    if d1 < d2:
        near = (int(round(x1)), int(round(y1)))
        far = (int(round(x2)), int(round(y2)))
    else:
        near = (int(round(x2)), int(round(y2)))
        far = (int(round(x1)), int(round(y1)))

    radial_vec = (far[0] - center[0], far[1] - center[1])
    if radial_vec == (0, 0):
        return None
    radial_norm = float(np.hypot(*radial_vec))
    dir_norm = float(length)
    alignment = abs((dx * radial_vec[0] + dy * radial_vec[1]) / (dir_norm * radial_norm + 1e-6))

    center_bonus = max(0.0, inner * 1.55 - near_dist)
    far_bonus = max(0.0, far_dist - outer * 0.5)
    score = (
        length
        + 1.8 * center_bonus
        + 1.15 * far_bonus
        + 55.0 * alignment
        - 1.5 * line_distance
    )

    dx_far = far[0] - center[0]
    dy_far = center[1] - far[1]
    angle_deg = (np.degrees(np.arctan2(dy_far, dx_far)) + 360.0) % 360.0

    return score, near, far, angle_deg


def _detect_pointer_with_lsd(
    enhanced: np.ndarray,
    mask: np.ndarray,
    center: tuple[int, int],
    inner: float,
    outer: float,
) -> tuple[dict, np.ndarray] | None:
    if not hasattr(cv2, "createLineSegmentDetector"):
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)
    lsd_input = cv2.addWeighted(enhanced, 0.6, blackhat, 1.4, 0)
    masked_input = cv2.bitwise_and(lsd_input, mask)

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines, _, _, _ = lsd.detect(masked_input)
    if lines is None:
        return None

    best_data: dict | None = None
    best_score = 0.0
    for line in lines:
        x1, y1, x2, y2 = map(float, line[0])
        candidate = _score_pointer_candidate(center, inner, outer, (x1, y1), (x2, y2))
        if candidate is None:
            continue
        score, near, far, angle = candidate
        if score > best_score:
            best_score = score
            best_data = {
                "line": (
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2)),
                    int(round(y2)),
                ),
                "near": near,
                "far": far,
                "angle_deg": angle,
                "center": center,
                "score": score,
            }

    if best_data is None:
        return None

    needle_mask = np.zeros_like(enhanced, dtype=np.uint8)
    x1, y1, x2, y2 = best_data["line"]
    cv2.line(needle_mask, (x1, y1), (x2, y2), 255, 2)
    return best_data, needle_mask


def _detect_pointer_with_hough(
    blurred: np.ndarray,
    mask: np.ndarray,
    center: tuple[int, int],
    inner: float,
    outer: float,
) -> tuple[dict, np.ndarray] | None:
    best_data: dict | None = None
    best_score = 0.0
    best_edges = None

    parameter_sets = (
        (0.22, 0.62, 18),
        (0.3, 0.58, 22),
        (0.38, 0.55, 26),
        (0.46, 0.5, 30),
    )

    for sigma, min_length_factor, max_gap in parameter_sets:
        edges = auto_canny(blurred, sigma=sigma)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        refined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        refined = cv2.medianBlur(refined, 3)
        masked_edges = cv2.bitwise_and(refined, mask)

        min_line_length = int(outer * min_length_factor)
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi / 180,
            threshold=38,
            minLineLength=max(10, min_line_length),
            maxLineGap=max_gap,
        )

        if lines is None:
            continue

        for line in lines:
            x1, y1, x2, y2 = map(float, line[0])
            candidate = _score_pointer_candidate(center, inner, outer, (x1, y1), (x2, y2))
            if candidate is None:
                continue
            score, near, far, angle = candidate
            if score > best_score:
                best_score = score
                best_edges = masked_edges
                best_data = {
                    "line": (
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                    ),
                    "near": near,
                    "far": far,
                    "angle_deg": angle,
                    "center": center,
                    "score": score,
                }

        if best_data is not None and best_score > outer:
            break

    if best_data is None:
        return None

    needle_mask = best_edges if best_edges is not None else np.zeros_like(mask, dtype=np.uint8)
    x1, y1, x2, y2 = best_data["line"]
    overlay = np.zeros_like(mask, dtype=np.uint8)
    cv2.line(overlay, (x1, y1), (x2, y2), 255, 2)
    needle_mask = cv2.bitwise_or(needle_mask, overlay)
    return best_data, needle_mask


def _detect_pointer_with_polar(
    enhanced: np.ndarray,
    center: tuple[int, int],
    inner: float,
    outer: float,
) -> tuple[dict, np.ndarray] | None:
    if outer <= 0:
        return None

    angle_bins = 720
    radius = int(max(outer, 1))
    polar = cv2.warpPolar(
        enhanced,
        (angle_bins, radius),
        center,
        radius,
        cv2.WARP_POLAR_LINEAR,
    )
    if polar is None or polar.size == 0:
        return None

    start = max(0, int(inner * 0.6))
    band = polar[start:radius]
    if band.size == 0:
        return None

    band = cv2.GaussianBlur(band, (1, 5), 0)
    mean_profile = band.mean(axis=0)
    min_profile = band.min(axis=0)
    max_profile = band.max(axis=0)

    dark_response = mean_profile - min_profile
    bright_response = max_profile - mean_profile
    combined = np.minimum(dark_response, bright_response)

    idx = int(np.argmin(combined))
    contrast = float(combined.max() - combined.min())
    if contrast < 6.0:
        return None

    angle_deg = (idx / angle_bins) * 360.0
    rad = np.deg2rad(angle_deg)
    near_radius = inner * 0.65
    far_radius = outer * 0.92
    line = (
        center[0] + int(round(np.cos(rad) * near_radius)),
        center[1] - int(round(np.sin(rad) * near_radius)),
        center[0] + int(round(np.cos(rad) * far_radius)),
        center[1] - int(round(np.sin(rad) * far_radius)),
    )

    candidate = _score_pointer_candidate(
        center,
        inner,
        outer,
        (float(line[0]), float(line[1])),
        (float(line[2]), float(line[3])),
    )
    if candidate is None:
        return None

    score, near, far, refined_angle = candidate
    score += contrast * 0.5

    data = {
        "line": line,
        "near": near,
        "far": far,
        "angle_deg": refined_angle,
        "center": center,
        "score": score,
    }

    mask = np.zeros_like(enhanced, dtype=np.uint8)
    cv2.line(mask, (line[0], line[1]), (line[2], line[3]), 255, 2)
    return data, mask


def detect_pointer(rectified: np.ndarray) -> tuple[dict | None, np.ndarray]:
    h, w = rectified.shape[:2]
    center = (w // 2, h // 2)
    max_radius = min(center)

    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

    mask = np.zeros_like(gray, dtype=np.uint8)
    outer = int(max_radius * 0.95)
    inner = int(max_radius * 0.2)
    cv2.circle(mask, center, outer, 255, -1)
    cv2.circle(mask, center, inner, 0, -1)

    lsd_result = _detect_pointer_with_lsd(enhanced, mask, center, inner, outer)
    if lsd_result is not None:
        return lsd_result

    hough_result = _detect_pointer_with_hough(blur, mask, center, inner, outer)
    if hough_result is not None:
        return hough_result

    polar_result = _detect_pointer_with_polar(enhanced, center, inner, outer)
    if polar_result is not None:
        return polar_result

    return None, np.zeros_like(gray)


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


def annotate_outputs(
    original: np.ndarray,
    ellipse: tuple | None,
    pointer: dict | None,
    angle: float | None,
    value: float | None,
) -> np.ndarray:
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


def process_image(
    image_path: str,
    output_paths: OutputPaths | None = None,
    calibrations: dict[str, GaugeCalibration] | None = None,
    write_outputs: bool = True,
) -> tuple[float | None, float | None]:
    image = cv2.imread(image_path)
    if image is None:
        print(f"No se pudo leer {image_path}")
        return None, None

    if calibrations is None:
        calibrations = GAUGE_CALIBRATIONS

    if write_outputs and output_paths is None:
        output_paths = OutputPaths.from_base(DEFAULT_OUTPUT_DIR)
        output_paths.ensure()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ellipse, edges = find_ellipse(gray)

    if ellipse is None:
        print(f"No se detectó elipse en {image_path}")
        if write_outputs and output_paths is not None:
            base = os.path.basename(image_path)
            cv2.imwrite(os.path.join(output_paths.annotated, base), image)
            cv2.imwrite(os.path.join(output_paths.mask, base), edges)
        return None, None

    rectified, _ = rectify_gauge(image, ellipse)
    if rectified is None:
        print(f"{os.path.basename(image_path)}: no se pudo recortar el gauge")
        return None, None

    pointer, needle_mask = detect_pointer(rectified)

    angle_deg = pointer["angle_deg"] if pointer else None
    value = None
    calib = calibrations.get(os.path.basename(image_path))
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

    if write_outputs and output_paths is not None:
        cv2.imwrite(os.path.join(output_paths.annotated, base_name), annotated_original)
        cv2.imwrite(os.path.join(output_paths.rectified, base_name), annotated_rectified)
        cv2.imwrite(os.path.join(output_paths.mask, base_name), needle_mask)

    if angle_deg is not None:
        message = f"{base_name}: ángulo {angle_deg:0.1f}°"
        if value is not None:
            message += f", valor {value:0.2f}"
        print(message)
    else:
        print(f"{base_name}: no se detectó aguja")

    return angle_deg, value


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detector heurístico de diales analógicos",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help=(
            "Rutas de imágenes, directorios o patrones glob. Si se omite, se procesan "
            "los ejemplos incluidos en samples/."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio donde guardar anotaciones y máscaras (por defecto: output/).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="No guardar imágenes ni máscaras procesadas, solo mostrar lecturas por consola.",
    )
    parser.add_argument(
        "--calibration",
        "-c",
        action="append",
        help=(
            "Ruta a un archivo JSON con calibraciones adicionales. Se puede repetir; "
            "las últimas entradas sobrescriben a las anteriores."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)

    image_paths = resolve_image_inputs(args.inputs, DEFAULT_SAMPLES_DIR)
    if not image_paths:
        if args.inputs:
            print("No se encontraron imágenes que coincidan con los parámetros proporcionados")
        else:
            print(f"No se encontraron imágenes en {DEFAULT_SAMPLES_DIR}")
        return 1

    write_outputs = not args.no_write
    output_paths = None
    if write_outputs:
        output_base = os.path.abspath(os.path.expanduser(args.output))
        output_paths = OutputPaths.from_base(output_base)
        output_paths.ensure()

    calibrations = dict(GAUGE_CALIBRATIONS)
    if args.calibration:
        for calibration_file in args.calibration:
            try:
                calibrations.update(
                    load_calibrations_from_file(os.path.expanduser(calibration_file))
                )
            except (OSError, ValueError) as exc:
                print(f"No se pudo cargar la calibración '{calibration_file}': {exc}")
                return 1

    for image_path in image_paths:
        process_image(
            image_path,
            output_paths=output_paths,
            calibrations=calibrations,
            write_outputs=write_outputs,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
