import os
import gc
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
import seaborn as sns
from sklearn.metrics import confusion_matrix

def crop_hand_simple(img_gray, output_size=64):
    h, w = img_gray.shape

    # Adaptive threshold handles varied lighting better than a fixed value
    thresh = cv2.adaptiveThreshold(
        img_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Morphological closing fills small gaps in the hand region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 1000]

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(c)
        pad = int(max(cw, ch) * 0.2)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + cw + pad), min(h, y + ch + pad)
        cropped = img_gray[y1:y2, x1:x2]
    else:
        # Warn instead of silently producing a bad crop
        print("No contour found — falling back to center crop")
        margin_y, margin_x = int(h * 0.2), int(w * 0.2)
        cropped = img_gray[margin_y:h - margin_y, margin_x:w - margin_x]

    # Square-pad with black to preserve aspect ratio
    ch, cw = cropped.shape
    size = max(ch, cw)
    canvas = np.zeros((size, size), dtype=np.uint8)
    canvas[(size - ch) // 2:(size - ch) // 2 + ch,
           (size - cw) // 2:(size - cw) // 2 + cw] = cropped

    # Resize to a fixed output resolution
    return cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_AREA)

def build_cropped_dataset(src, dst):
    if os.path.exists(dst):
        print(f"Cropped dataset already exists at {dst}, skipping.")
        return

    print(f"Building cropped dataset: {src} → {dst}")
    total, failed = 0, 0 # Counters for reporting progress and issues

    # Process each label folder in the source directory
    for label in sorted(os.listdir(src)):
        src_dir = os.path.join(src, label)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(dst, label)
        os.makedirs(dst_dir, exist_ok=True)

        # Process each image in the label folder
        for fname in os.listdir(src_dir):
            src_path = os.path.join(src_dir, fname)
            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Could not read {src_path}, skipping.")
                failed += 1
                continue
            cropped = crop_hand_simple(img)
            cv2.imwrite(os.path.join(dst_dir, fname), cropped) # Save cropped image
            total += 1

    print(f"Cropping complete: {total} saved, {failed} skipped.")

# Resize MNIST images to 64×64 and re-cast to uint8 for consistency with ASL dataset
def resize_and_cast(x, y):
    x = tf.image.resize(x, [64, 64])
    x = tf.clip_by_value(x, 0, 255)
    x = tf.cast(x, tf.uint8)
    return x, y

# Randomly apply black borders to simulate inference where borders may be added for centering
def random_black_background(x, y):
    def apply_mask():
        mask = tf.cast(x > 0.2, tf.float32)
        return x * mask, y
    def keep_original():
        return x, y
    return tf.cond(tf.random.uniform(()) > 0.5, apply_mask, keep_original)

# augment MNIST by blurring to encourage robustness
def augment_mnist(x, y):
    x = tf.nn.avg_pool2d(x, ksize=2, strides=1, padding='SAME')
    return x, y


def SLR():
    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    ASL_CROP  = os.path.join(BASE_DIR, 'data', 'asl_cropped')

    # Load ASL dataset

    asl_train = tf.keras.utils.image_dataset_from_directory(ASL_CROP,
        subset="training",
        image_size=(64, 64),
        color_mode="grayscale",
        batch_size=64,
        label_mode="categorical",
        validation_split=0.2,
        seed=0
    )

    asl_test = tf.keras.utils.image_dataset_from_directory(ASL_CROP,
        subset="validation",
        image_size=(64, 64),
        color_mode="grayscale",
        batch_size=64,
        label_mode="categorical",
        validation_split=0.2,
        seed=0
    )

    # Normalize to [0, 1] and apply random black borders to match inference
    asl_train = (asl_train
                 .map(lambda x, y: (x / 255.0, y))
                 .map(random_black_background))
    asl_test  = (asl_test
                 .map(lambda x, y: (x / 255.0, y)))

    # Load MNIST dataset

    train_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'sign_mnist_train.csv'))
    test_df  = pd.read_csv(os.path.join(BASE_DIR, 'data', 'sign_mnist_test.csv'))

    def remap_mnist_label(label):
        # MNIST skips J (9) and Z (25); shift labels above 9 down by 1
        # to produce a dense 0–23 range matching the ASL label space
        return label - 1 if label > 9 else label

    remapped_train = np.array([remap_mnist_label(l) for l in train_df["label"].values])
    remapped_test  = np.array([remap_mnist_label(l) for l in test_df["label"].values])

    # One-hot encode remapped labels for 24 classes
    mnist_train_labels = tf.keras.utils.to_categorical(remapped_train, 24).astype(np.float32)
    mnist_test_labels  = tf.keras.utils.to_categorical(remapped_test, 24).astype(np.float32)

    # Reshape pixel columns back into (28, 28, 1) image tensors
    mnist_train = train_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1).astype(np.float32)
    mnist_test  = test_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1).astype(np.float32)

    # Free DataFrames and remapped arrays
    del train_df, test_df, remapped_train, remapped_test
    gc.collect()

    def make_mnist_ds(x, y):
        # Resize to 64×64, cast to uint8, renormalize, then apply avg-pool augmentation
        return (tf.data.Dataset.from_tensor_slices((x, y))
                .map(resize_and_cast)
                .batch(64)
                .map(lambda xi, yi: (tf.cast(xi, tf.float32) / 255.0, yi))
                .map(augment_mnist))

    mnist_train_ds = make_mnist_ds(mnist_train, mnist_train_labels)
    mnist_test_ds  = make_mnist_ds(mnist_test,  mnist_test_labels)

    # Free raw MNIST arrays and labels
    del mnist_train, mnist_test, mnist_train_labels, mnist_test_labels
    gc.collect()

    # Combine ASL and MNIST datasets for training and evaluation

    # Unbatch both sources, shuffle, re-batch, and prefetch for training
    train_ds = (asl_train.unbatch()
                .shuffle(5000)
                .concatenate(mnist_train_ds.unbatch().shuffle(5000))
                .shuffle(5000)
                .batch(64)
                .prefetch(tf.data.AUTOTUNE))

    # Unbatch for consistent concatenation; no shuffle needed for evaluation
    test_ds  = (asl_test.unbatch()
                .concatenate(mnist_test_ds.unbatch())
                .batch(64)
                .prefetch(tf.data.AUTOTUNE))

    # Model

    # Define fixed hand-crafted filters (commented out — learned filters used instead)

    # Gaussian blur 3×3
    #gaussian_kernel = np.array([[1,2,1],
    #                            [2,4,2],
    #                            [1,2,1]], dtype=np.float32) / 16

    # Sobel X 3×3
    #sobel_x = np.array([[-1,0,1],
    #                    [-2,0,2],
    #                    [-1,0,1]], dtype=np.float32)

    # Sobel Y 3×3
    #sobel_y = np.array([[-1,-2,-1],
    #                    [0,0,0],
    #                    [1,2,1]], dtype=np.float32)

    # Stack them as separate filters
    # Conv2D expects (filter_height, filter_width, in_channels, out_channels)
    #fixed_kernels = np.stack([gaussian_kernel, sobel_x, sobel_y], axis=-1)
    #fixed_kernels = fixed_kernels[:, :, np.newaxis, :]  # shape (3,3,1,3)

    model = Sequential([
        # Data augmentation for improved generalization
        tf.keras.layers.RandomRotation(0.25, input_shape=(64, 64, 1)),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomTranslation(0.15, 0.15),
        tf.keras.layers.RandomBrightness(0.3, value_range=(0, 1)),
        tf.keras.layers.RandomContrast(0.5),
        tf.keras.layers.GaussianNoise(0.05),

        # Optional fixed-filter layer (not trainable) — disabled
        #Conv2D(3, (3,3), padding='same', use_bias=False,
        #       trainable=False, input_shape=(64, 64, 1)),

        # 32 filters, 3×3 kernel — learns low-level features (edges, corners)
        # padding='same' preserves spatial dimensions after convolution
        # 2x2 max pooling reduces spatial dimensions by half
        # Batch normalization after pooling helps stabilize and accelerate training
        Conv2D(32,  (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        MaxPooling2D(2, 2),

        # 64 filters, 3×3 kernel — learns mid-level features (combinations of edges)
        Conv2D(64,  (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        MaxPooling2D(2, 2),

        # 128 filters, 3×3 kernel — learns higher-level features (hand shapes)
        Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        MaxPooling2D(2, 2),

        # 256 filters, 3×3 kernel — learns complex features (whole hand configurations)
        Conv2D(256, (3,3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        MaxPooling2D(2, 2),

        # Global average pooling collapses spatial dimensions, leaving only the channel dimension
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),  # strong dropout to combat overfitting
        # Softmax output over 24 classes (A–Y, excluding J and Z)
        Dense(24, activation='softmax'),
    ])

    # Set fixed kernel weights if using the hand-crafted filter layer
    #model.layers[0].set_weights([fixed_kernels])

    # Compile with Adam optimizer and categorical cross-entropy

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks for early stopping and learning rate decay

    callbacks = [
        # Stop early if accuracy does not improve
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.001, restore_best_weights=True),
        # Halve the LR after loss does not improve
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
    ]


    # Letters for visualization and confusion matrix
    letters = list('ABCDEFGHIKLMNOPQRSTUVWXY')

    # Grid of sample images from each dataset for testing and visualization
    def save_sample_grid(dataset, name):
        images, labels = next(iter(dataset))
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            img   = images[i].numpy().squeeze()
            label = letters[tf.argmax(labels[i]).numpy()]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(label, fontsize=10)
            ax.axis('off')
        plt.suptitle(name, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, f'sample_{name}.png'), dpi=150)
        plt.close()

    save_sample_grid(asl_train,     'asl_train')
    save_sample_grid(mnist_train_ds,'mnist_train')
    save_sample_grid(train_ds,      'combined_train')

    # Train the model and collect the history

    history = model.fit(train_ds, epochs=50, validation_data=test_ds, callbacks=callbacks)

    # Loss and Accuracy plots

    for metric, title in [('loss', 'Loss'), ('accuracy', 'Accuracy')]:
        plt.figure()
        plt.plot(history.history[metric],          label=f'Training {title}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {title}')
        plt.xlabel('Epochs')
        plt.ylabel(title)
        plt.title(f'{title} vs Epochs')
        plt.legend()
        plt.savefig(os.path.join(BASE_DIR, f'{metric}_curve.png'), dpi=150)
        plt.show()

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Confusion Matrix

    y_true, y_pred = [], []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_true.extend(np.argmax(y_batch.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=letters, yticklabels=letters)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'), dpi=150)
    plt.show()

    # Save the model
    model.save(os.path.join(BASE_DIR, 'slr_model.keras'))
    print("Model saved to slr_model.keras")

    return model

if __name__ == "__main__":
    build_cropped_dataset(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'asl_alphabet_train', 'asl_alphabet_train'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'asl_cropped')
    )
    SLR()