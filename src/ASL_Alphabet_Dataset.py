import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import os
import random
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.patches import Rectangle

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Load CSV data
#train_df = pd.read_csv("data/sign_mnist/sign_mnist_train.csv")
#test_df = pd.read_csv("data/sign_mnist/sign_mnist_test.csv")

# The csv contains 785 columns, the first column is the label and the rest are pixel values.
# Each row is one image.
# The images are 28 x 28 pixels grayscale.
# Due to grayscale we divide by 255 to make the values of the pixels between 0 and 1 (normalization).
# Reshape creates 28 x 28 matrix for each row of 784 pixels.
# -1 in reshape means the total number of images is found dynamically, 1 means 1 dimension for grayscale
#X_train = train_df.drop("label", axis=1).values.reshape(-1,28,28,1) / 255.0
#y_train = tf.keras.utils.to_categorical(train_df["label"].values, 26)

#X_test = test_df.drop("label", axis=1).values.reshape(-1,28,28,1) / 255.0
#y_test = tf.keras.utils.to_categorical(test_df["label"].values, 26)


DATASET_DIR          = "data/ASL_kaggle/asl_alphabet_train"       # <-- root folder containing one subdir per letter
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
IMG_SIZE             = (64, 64)
TEST_SPLIT           = 0.2                 # 80% train, 20% test

# Sign Language mapping (0=A, 1=B, ..., 25=Z)
letters = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
]

def label_to_letter(label):
    return letters[label]
 
MAX_PER_CLASS = 2000

letter_dirs = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])
 
X, y = [], []
 
for folder in letter_dirs:
    folder_path = os.path.join(DATASET_DIR, folder)
    label_str = folder.upper()
 
    if label_str not in letters:
        print(f"  [SKIP] '{folder}' is not a valid letter folder")
        continue
 
    label_idx = letters.index(label_str)
 
    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    images = images[:MAX_PER_CLASS]  # cap per class
 
    loaded = 0
    for fname in images:
        img_path = os.path.join(folder_path, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
 
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, IMG_SIZE)
        X.append(resized)
        y.append(label_idx)
        loaded += 1
 
    print(f"  Loaded {loaded} images for '{label_str}'")
 
X = np.array(X, dtype=np.float32) / 255.0
y = np.array(y, dtype=np.int32)
 
print(f"\nTotal images loaded: {len(X)}")
print(f"Splitting into {int((1 - TEST_SPLIT) * 100)}% train / {int(TEST_SPLIT * 100)}% test...\n")
 
y_cat = tf.keras.utils.to_categorical(y, 26)
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=TEST_SPLIT, random_state=seed_value, stratify=y
)
 
del X, y, y_cat

print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")


# Define fixed filters

# Gaussian blur 3x3
gaussian_kernel = np.array([[1,2,1],
                            [2,4,2],
                            [1,2,1]], dtype=np.float32) / 16

# Sobel X 3x3
sobel_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]], dtype=np.float32)

# Sobel Y 3x3
sobel_y = np.array([[-1,-2,-1],
                    [0,0,0],
                    [1,2,1]], dtype=np.float32)

laplacian = np.array([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]], dtype=np.float32)

# Stack them as separate filters
# Conv2D expects (filter_height, filter_width, in_channels, out_channels)
fixed_kernels = np.stack([gaussian_kernel, sobel_x, sobel_y, laplacian], axis=-1)
fixed_kernels = fixed_kernels[:, :, np.newaxis, :]  # shape (3,3,1,4)
fixed_kernels = np.repeat(fixed_kernels, 3, axis=2)       # shape (3,3,3,4)

# CNN
model = Sequential([
    # Fixed filters layer (not trainable)
    Conv2D(4, (3,3), padding='same', use_bias=False, 
           trainable=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    # Learned convolution layers
    # 32 filters with a 3x3 kernel for learning low level features
    # padding = same keeps the original image size
    # MaxPooling does pooling by choosing the maximum value in a 2x2 pixel matrix
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    
    # 64 filters with a 3x3 kernel for learning high level features
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),

    # Neural Network with one hidden layer of 128 neurons and 25 classes in the output layer
    Flatten(),
    Dense(256, activation='relu'),
    # Dropout layer randomly disables neurons during training to prevent memorization
    tf.keras.layers.Dropout(0.7), 
    Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5), 
    # The output is one-hot encoded as the output vector is always size 26 but only one index is true while the rest are false
    Dense(26, activation='softmax')
])

# Set the fixed kernels
model.layers[0].set_weights([fixed_kernels])

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=15, batch_size=64,
                    validation_data=(X_test, y_test))

# model.save("slr_kaggle_model.keras")   # saves architecture + weights + optimizer state
# print("Model saved.")

# ── Visualise fixed filter outputs on one training image ─────────────────────

sample = X_train[0:1]   # shape (1, IMG_SIZE[0], IMG_SIZE[1], 1)

filter_model = tf.keras.Model(inputs=model.input, outputs=model.layers[0].output)
filter_outputs = filter_model.predict(sample, verbose=0)   # shape (1, IMG_SIZE[0], IMG_SIZE[1], 3)

filter_names = ["Gaussian Blur", "Sobel X", "Sobel Y", "laplacian"]

plt.figure(figsize=(15, 3))

# Original image
plt.subplot(1, 5, 1)
plt.imshow(sample[0]) # .reshape(IMG_SIZE[0], IMG_SIZE[1]), cmap='gray')
plt.title("Original")
plt.axis("off")

# One output per filter
for i, name in enumerate(filter_names):
    plt.subplot(1, 5, i + 2)
    plt.imshow(filter_outputs[0, :, :, i], cmap='gray')
    plt.title(name)
    plt.axis("off")

plt.suptitle("Fixed Filter Outputs", fontsize=13)
plt.tight_layout()
plt.show()

# Evaluate
# Plot of loss history
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend()

plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot of confusion matrix for test dataset
y_pred_val = np.argmax(model.predict(X_test), axis=1)
y_true_val = np.argmax(y_test, axis=1)
cm_val = confusion_matrix(y_true_val, y_pred_val)
 
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=letters)
disp.plot(ax=plt.gca(), colorbar=True, xticks_rotation=45)
plt.title("Confusion Matrix — Validation Split")
plt.tight_layout()
plt.show()

# Plot of model accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.legend()

plt.show()

# Predict twelve images from X_test

# first 12 images
num_images = 12
X_subset = X_test[:num_images] 
y_subset = y_test[:num_images]  

predictions = model.predict(X_subset)
predicted_labels = np.argmax(predictions, axis=1)
predicted_letters = [label_to_letter(l) for l in predicted_labels]

true_labels = np.argmax(y_subset, axis=1) 
true_letters = [label_to_letter(l) for l in true_labels]
print("Predicted letters:", predicted_letters)
print("True letters:     ", true_letters)

plt.figure(figsize=(8, 6))
for i in range(num_images):
    ax = plt.subplot(3, 4, i+1)
    ax.imshow(X_subset[i])
    ax.add_patch(Rectangle((0, 0), 1, 0.18, transform=ax.transAxes,
                            facecolor='white', alpha=0.85, zorder=2))
    ax.text(0.5, 0.09, f"Pred: {predicted_letters[i]}  True: {true_letters[i]}",
            transform=ax.transAxes, fontsize=8, color='black',
            ha='center', va='center', zorder=3)
    ax.axis('off')
plt.subplots_adjust(wspace=0.01, hspace=0.05)
plt.show()

# GradCam on validation images
CONV_LAYERS = ["conv2d_1", "conv2d_2", "conv2d_3"]
NUM_SAMPLES = 12  # how many val images to visualize

def make_gradcam_heatmap(img_array, true_class, model, last_conv_layer_name):
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.cast(img_array, tf.float32))
        loss = predictions[:, true_class]  # use true class instead of predicted

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(img_rgb_float, heatmap):
    h, w = img_rgb_float.shape[:2]
    img_uint8 = (img_rgb_float * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * img_uint8 + 0.4 * heatmap_colored).astype(np.uint8)
    return overlay

# Sample from validation set
indices = np.random.choice(len(X_test), NUM_SAMPLES, replace=False)
X_sample = X_test[indices]
y_sample = y_test[indices]

preds = model.predict(X_sample)
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(y_sample, axis=1)

cols = 4
rows = (NUM_SAMPLES + cols - 1) // cols

for layer_name in CONV_LAYERS:
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    correct_count = 0
    for i in range(NUM_SAMPLES):
        img_array = X_sample[i:i+1]
        heatmap = make_gradcam_heatmap(img_array, true_labels[i], model, layer_name)
        overlay = overlay_gradcam(X_sample[i], heatmap)

        pred_letter = label_to_letter(pred_labels[i])
        true_letter = label_to_letter(true_labels[i])
        confidence = float(preds[i][pred_labels[i]])
        correct = pred_letter == true_letter
        if correct:
            correct_count += 1

        axes[i].imshow(overlay)
        axes[i].set_title(f"True: {true_letter}  Pred: {pred_letter}\n{confidence*100:.1f}%",
                          color="green" if correct else "red")
        axes[i].axis("off")

    for j in range(NUM_SAMPLES, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"GradCAM [{layer_name}] — {correct_count}/{NUM_SAMPLES} correct",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

# External dataset run

IMAGE_DIR = "data/asl_dataset"   # <-- change this, subdirs should be named A, B, C ...
SAMPLES_PER_LETTER = 1              # <-- how many random images to pick per letter
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

def preprocess_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(gray, (IMG_SIZE[1], IMG_SIZE[0]))
    return resized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 3) / 255.0

# Sample images from subdirectories

# Each subdir name is treated as the true label (e.g. "A", "B", ...)
letter_dirs = {
    d: os.path.join(IMAGE_DIR, d)
    for d in sorted(os.listdir(IMAGE_DIR))
    if os.path.isdir(os.path.join(IMAGE_DIR, d))
    and d.upper() not in ('J', 'Z')
}

print(f"Found {len(letter_dirs)} letter folder(s): {', '.join(letter_dirs.keys())}\n")

sampled = []   # (true_letter, img_path)
for letter, folder in letter_dirs.items():
    all_images = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    if not all_images:
        print(f"  [EMPTY] No images in folder '{letter}'")
        continue
    chosen = random.sample(all_images, min(SAMPLES_PER_LETTER, len(all_images)))
    sampled.extend((letter, path) for path in chosen)

print(f"Sampled {len(sampled)} image(s) total. Running predictions...\n")

# Run predictions

results = []   # (true_letter, predicted_letter, confidence, img)
for true_letter, path in sampled:
    img = cv2.imread(path)
    if img is None:
        print(f"  [SKIP] Could not read: {path}")
        continue

    if img is None:
        print(f"  [NO HAND] {os.path.basename(path)}  (true: {true_letter})")
        continue

    preds = model.predict(preprocess_img(img), verbose=0)[0]
    best_idx = int(np.argmax(preds))
    confidence = float(preds[best_idx])
    pred_letter = label_to_letter(best_idx)
    results.append((true_letter, pred_letter, confidence, img))

    correct = "✓" if pred_letter == true_letter else "✗"
    print(f"  {correct}  true: {true_letter}  pred: {pred_letter}  ({confidence*100:.1f}%)  {os.path.basename(path)}")

# Summary

correct_count = sum(1 for t, p, _, __ in results if t == p)
print(f"\nAccuracy on sample: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")

# Plot

n = len(results)
cols = min(SAMPLES_PER_LETTER * 6, n)
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(10, 8))
axes = np.array(axes).reshape(-1)

for i, (true_letter, pred_letter, confidence, img) in enumerate(results):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    correct = pred_letter == true_letter
    color = "green" if correct else "red"
    axes[i].imshow(img_rgb)
    # Full-width background bar
    axes[i].add_patch(Rectangle((0, 0), 1, 0.15, transform=axes[i].transAxes,
                                 facecolor='white', alpha=0.5, zorder=2))
    axes[i].text(0.5, 0.12, f"True: {true_letter}  Pred: {pred_letter}  {confidence*100:.1f}%",
                 transform=axes[i].transAxes, fontsize=8, color=color,
                 ha='center', va='top', zorder=3)
    axes[i].axis("off")

for j in range(len(results), len(axes)):
    axes[j].axis("off")

plt.suptitle(f"Sample Predictions — {correct_count}/{len(results)} correct", fontsize=14)
plt.tight_layout()
plt.show()

# Confusion matrix on external dataset

true_letters_cm = [t for t, p, _, __ in results]
pred_letters_cm = [p for t, p, _, __ in results]
 
present_letters = sorted(set(true_letters_cm) | set(pred_letters_cm))
cm_ext = confusion_matrix(true_letters_cm, pred_letters_cm, labels=present_letters)
 
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_ext, display_labels=present_letters)
disp.plot(ax=plt.gca(), colorbar=True, xticks_rotation=45)
plt.title("Confusion Matrix — External Dataset (Real World)")
plt.tight_layout()
plt.show()