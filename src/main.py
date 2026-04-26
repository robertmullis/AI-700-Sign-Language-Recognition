import os
import sys
import tensorflow as tf
import numpy as np
import mediapipe as mp
import cv2

LETTERS = list('ABCDEFGHIKLMNOPQRSTUVWXY')

def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"cv2.imread failed to decode: {path}")
    return img


def detect_hand_bounds(img: np.ndarray) -> tuple[int, int, int, int]:
    # Convert BGR to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_hands = mp.solutions.hands

    # Use MediaPipe to detect hand landmarks
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
    ) as hands:
        results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        raise RuntimeError("No hand detected in the image.")

    # Get bounding box from landmarks
    h_img, w_img = img.shape[:2]
    landmarks = results.multi_hand_landmarks[0].landmark

    x = [int(lm.x * w_img) for lm in landmarks]
    y = [int(lm.y * h_img) for lm in landmarks]

    return min(y), max(y), min(x), max(x)

def crop_hand(img_gray: np.ndarray, upper: int, lower: int,
              left: int, right: int, pad_frac: float = 0.2) -> np.ndarray:
    
    # Add padding to the bounding box
    h_img, w_img = img_gray.shape[:2]
    size = max(lower - upper, right - left)
    padding = int(size * pad_frac)

    # Ensure padding doesn't go out of bounds
    upper = max(0, upper - padding)
    lower = min(h_img, lower + padding)
    left  = max(0, left  - padding)
    right = min(w_img, right + padding)

    cropped = img_gray[upper:lower, left:right]

    if cropped.size == 0:
        raise ValueError("Crop is empty — landmarks may be out of frame.")

    # Square-pad with black to preserve aspect ratio
    h_c, w_c = cropped.shape[:2]
    max_dim   = max(h_c, w_c)
    pad_top   = (max_dim - h_c) // 2
    pad_bot   = max_dim - h_c - pad_top
    pad_left  = (max_dim - w_c) // 2
    pad_right = max_dim - w_c - pad_left

    return cv2.copyMakeBorder(
        cropped, pad_top, pad_bot, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0,
    )


def preprocess_for_model(crop: np.ndarray, target_size: int = 64) -> tf.Tensor:
    # Convert to float32, resize, normalise to [0,1], and add batch dimension
    tensor = tf.image.resize(crop[..., np.newaxis], [target_size, target_size])
    tensor = tensor / 255.0

    return tf.expand_dims(tensor, 0)


def predict(model, tensor: tf.Tensor) -> tuple[str, float, list[tuple[str, float]]]:

    # Get probabilities for each letter
    probs = model.predict(tensor, verbose=0)[0]
    top3_indices = tf.argsort(probs, direction='DESCENDING')[:3].numpy() # Sort and get top 3 letters

    predicted_idx    = top3_indices[0]
    predicted_letter = LETTERS[predicted_idx]
    confidence       = float(probs[predicted_idx])
    top3             = [(LETTERS[i], float(probs[i])) for i in top3_indices]

    return predicted_letter, confidence, top3


def main():
    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "slr_model.keras")
    image_path = os.path.join(BASE_DIR, "data", "test_images", "IMG_7239.JPG")
    debug_path = os.path.join(BASE_DIR, "data", "test_images", "cropped.jpg")

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load Image
    img = load_image(image_path)

    # Detect hand and get bounding box
    upper, lower, left, right = detect_hand_bounds(img)

    # Preprocess: crop, pad, resize, normalise
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropped  = crop_hand(img_gray, upper, lower, left, right)
    tensor   = preprocess_for_model(cropped)

    # Predict
    predicted_letter, confidence, top3 = predict(model, tensor)

    # Save crop for debugging
    cv2.imwrite(debug_path, cropped)

    # Output results

    print(f"Predicted: {predicted_letter} ({confidence:.1%} confidence)")
    print("Top 3:")
    for i, (letter, prob) in enumerate(top3, 1):
        print(f"  #{i}: {letter} ({prob:.1%})")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)