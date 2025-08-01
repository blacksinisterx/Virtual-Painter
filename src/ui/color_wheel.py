# Color Wheel UI Component

import numpy as np
import cv2
import math

def hsv_to_bgr(h, s, v):
    """Convert HSV to BGR color."""
    hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(map(int, bgr[0][0]))

def get_color_from_wheel_position(x, y, center_x, center_y, radius):
    """Get color based on position in color wheel."""
    dx = x - center_x
    dy = y - center_y
    distance = math.sqrt(dx * dx + dy * dy)
    if distance > radius:
        return None
    angle = math.atan2(dy, dx)
    hue = int((angle + math.pi) * 180 / (2 * math.pi)) % 180
    saturation = int((distance / radius) * 255)
    value = 255
    return hsv_to_bgr(hue, saturation, value)

def create_color_wheel_image(radius):
    """Create enhanced color wheel with smooth gradients."""
    size = radius * 2 + 1
    wheel_img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            dx = x - radius
            dy = y - radius
            distance = math.sqrt(dx * dx + dy * dy)
            if distance <= radius:
                angle = math.atan2(dy, dx)
                hue = int((angle + math.pi) * 180 / (2 * math.pi)) % 180
                saturation = int((distance / radius) * 255)
                value = 255
                color = hsv_to_bgr(hue, saturation, value)
                wheel_img[y, x] = color
    return wheel_img

def overlay_color_wheel(frame, center_x, center_y, radius, wheel_img, alpha=0.9):
    """Enhanced color wheel overlay."""
    y1 = center_y - radius
    y2 = center_y + radius + 1
    x1 = center_x - radius
    x2 = center_x + radius + 1

    h, w, _ = frame.shape
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return

    roi = frame[y1:y2, x1:x2].copy()
    mask = np.any(wheel_img != [0, 0, 0], axis=2)

    blended = roi.copy()
    blended[mask] = wheel_img[mask]
    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, blended, alpha, 0)