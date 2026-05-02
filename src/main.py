"""
Live ASL Sign Language Recognition
-----------------------------------
Controls:
  Q / ESC — quit
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from collections import deque

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH       = "model_phase3.keras"       # path to your .keras model
LANDMARKER_PATH  = "hand_landmarker.task"     # path to mediapipe task file
IMG_SIZE         = 64
CONFIDENCE_THRESH = 0.5
SMOOTHING_FRAMES  = 10   # average predictions over this many frames to reduce flicker
# ─────────────────────────────────────────────────────────────────────────────

LETTERS = list('ABCDEFGHIKLMNOPQRSTUVWXY')


def build_detector():
    base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_hand_crop(frame_rgb: np.ndarray, detector, output_size: int = IMG_SIZE):
    """
    Detect hand in frame_rgb, return (crop_rgb, bbox) or (None, None) if no hand found.
    bbox = (x1, y1, x2, y2) in original frame coordinates — used to draw the box.
    """
    h, w = frame_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None, None

    landmarks = result.hand_landmarks[0]
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    hand_w = max_x - min_x
    hand_h = max_y - min_y
    side = int(max(hand_w, hand_h) * 2.0)

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    x1 = max(0, center_x - side // 2)
    y1 = max(0, center_y - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None, None

    crop_resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return crop_resized, (x1, y1, x2, y2)


def preprocess_for_model(crop_rgb: np.ndarray) -> tf.Tensor:
    tensor = crop_rgb.astype(np.float32) / 255.0
    return tf.expand_dims(tensor, axis=0)


def draw_overlay(frame, bbox, letter, confidence, top3, low_confidence):
    h, w = frame.shape[:2]

    # Bounding box around hand
    if bbox:
        x1, y1, x2, y2 = bbox
        color = (0, 200, 80) if not low_confidence else (0, 140, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Top prediction — large, top left
    if letter:
        conf_pct = f"{confidence:.0%}"
        label_color = (0, 200, 80) if not low_confidence else (0, 140, 255)

        cv2.putText(frame, letter, (16, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, label_color, 5, cv2.LINE_AA)
        cv2.putText(frame, conf_pct, (16, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2, cv2.LINE_AA)

        if low_confidence:
            cv2.putText(frame, "Low confidence", (16, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 1, cv2.LINE_AA)

        # Top 3 breakdown — bottom left
        for i, (l, p) in enumerate(top3):
            bar_w = int(p * 160)
            bar_y = h - 80 + i * 24
            cv2.rectangle(frame, (12, bar_y), (12 + bar_w, bar_y + 16),
                          (60, 60, 60) if i > 0 else label_color, -1)
            cv2.putText(frame, f"{l}  {p:.0%}", (16, bar_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No hand detected", (16, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2, cv2.LINE_AA)

    # Quit hint
    cv2.putText(frame, "Q / ESC to quit", (w - 160, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

    return frame


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(LANDMARKER_PATH):
        raise FileNotFoundError(f"hand_landmarker.task not found: {LANDMARKER_PATH}")

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Building MediaPipe detector...")
    detector = build_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try changing VideoCapture(0) to VideoCapture(1).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Rolling buffer to smooth predictions over recent frames
    prob_buffer = deque(maxlen=SMOOTHING_FRAMES)

    print("Live inference running. Press Q or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)  # mirror so it feels natural
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        crop, bbox = extract_hand_crop(frame_rgb, detector)

        letter = None
        confidence = 0.0
        top3 = []
        low_confidence = False

        if crop is not None:
            tensor = preprocess_for_model(crop)
            probs = model.predict(tensor, verbose=0)[0]
            prob_buffer.append(probs)

            # Average probabilities across recent frames
            avg_probs = np.mean(prob_buffer, axis=0)
            top3_idx = np.argsort(avg_probs)[-3:][::-1]

            letter = LETTERS[top3_idx[0]]
            confidence = float(avg_probs[top3_idx[0]])
            top3 = [(LETTERS[i], float(avg_probs[i])) for i in top3_idx]
            low_confidence = confidence < CONFIDENCE_THRESH
        else:
            prob_buffer.clear()  # reset smoothing when hand leaves frame

        display = draw_overlay(frame, bbox, letter, confidence, top3, low_confidence)
        cv2.imshow("ASL Live Inference", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Closed.")


if __name__ == "__main__":
    main()