import cv2
import numpy as np

def draw_rounded_rectangle(frame, top_left, bottom_right, color, thickness=-1, radius=10):
    """Draw a rounded rectangle with better precision"""
    x1, y1 = top_left
    x2, y2 = bottom_right

    if x2 <= x1 or y2 <= y1:
        return

    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    if thickness == -1:  # Filled
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1)
    else:  # Outline
        cv2.line(frame, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(frame, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(frame, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(frame, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)

def draw_glass_effect(frame, top_left, bottom_right, alpha=0.2):
    """Create a glass morphism effect"""
    overlay = frame.copy()
    draw_rounded_rectangle(overlay, top_left, bottom_right, (255, 255, 255), -1, 15)
    cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, frame)

def draw_modern_shadow(frame, top_left, bottom_right, intensity=0.3, offset=4):
    """Draw a modern, soft shadow"""
    x1, y1 = top_left
    x2, y2 = bottom_right
    shadow_overlay = frame.copy()

    for i in range(3):
        shadow_alpha = intensity * (0.5 - i * 0.15)
        shadow_offset = offset + i
        draw_rounded_rectangle(shadow_overlay,
                               (x1 + shadow_offset, y1 + shadow_offset),
                               (x2 + shadow_offset, y2 + shadow_offset),
                               (0, 0, 0), -1, 15)
        cv2.addWeighted(frame, 1 - shadow_alpha, shadow_overlay, shadow_alpha, 0, frame)
