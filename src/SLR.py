import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load CSV data
train_df = pd.read_csv("../data/sign_mnist/sign_mnist_train.csv")
test_df = pd.read_csv("../data/sign_mnist/sign_mnist_test.csv")

# The csv contains 785 columns, the first column is the label and the rest are pixel values.
# Each row is one image.
# The images are 28 x 28 pixels grayscale.
# Due to grayscale we divide by 255 to make the values of the pixels between 0 and 1 (normalization).
# Reshape creates 28 x 28 matrix for each row of 784 pixels.
# -1 in reshape means the total number of images is found dynamically, 1 means 1 dimension for grayscale
X_train = train_df.drop("label", axis=1).values.reshape(-1,28,28,1) / 255.0
y_train = tf.keras.utils.to_categorical(train_df["label"].values, 26)

X_test = test_df.drop("label", axis=1).values.reshape(-1,28,28,1) / 255.0
y_test = tf.keras.utils.to_categorical(test_df["label"].values, 26)

# Define fixed filters

# Gaussian blur 3x3
#gaussian_kernel = np.array([[1,2,1],
#                            [2,4,2],
#                            [1,2,1]], dtype=np.float32) / 16

# Sobel X 3x3
#sobel_x = np.array([[-1,0,1],
#                    [-2,0,2],
#                    [-1,0,1]], dtype=np.float32)

# Sobel Y 3x3
#sobel_y = np.array([[-1,-2,-1],
#                    [0,0,0],
#                    [1,2,1]], dtype=np.float32)

# Stack them as separate filters
# Conv2D expects (filter_height, filter_width, in_channels, out_channels)
#fixed_kernels = np.stack([gaussian_kernel, sobel_x, sobel_y], axis=-1)
#fixed_kernels = fixed_kernels[:, :, np.newaxis, :]  # shape (3,3,1,3)

# CNN
model = Sequential([
    # Fixed filters layer (not trainable)
    #Conv2D(3, (3,3), padding='same', use_bias=False, 
    #       trainable=False, input_shape=(28,28,1)),
    
    # Learned convolution layers
    # 32 filters with a 3x3 kernel for learning low level features
    # padding = same keeps the original image size
    # MaxPooling does pooling by choosing the maximum value in a 2x2 pixel matrix
    Conv2D(32, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    
    # 64 filters with a 3x3 kernel for learning high level features
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(2,2),
    
    # Neural Network with one hidden layer of 128 neurons and 25 classes in the output layer
    Flatten(),
    Dense(128, activation='relu'),
    # Dropout layer randomly disables neurons during training to prevent memorization
    tf.keras.layers.Dropout(0.7), 
    # The output is one-hot encoded as the output vector is always size 26 but only one index is true while the rest are false
    Dense(26, activation='softmax')
])

# Set the fixed kernels
#model.layers[0].set_weights([fixed_kernels])

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=15, batch_size=64,
                    validation_data=(X_test, y_test))

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

# Plot of model accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.legend()

plt.show()

# Predict twenty images from X_test

# Sign Language MNIST mapping (0=A, 1=B, ..., 25=Z)
letters = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
]

def label_to_letter(label):
    return letters[label]

# first 20 images
num_images = 20
X_subset = X_test[:num_images] 
y_subset = y_test[:num_images]  

predictions = model.predict(X_subset)
predicted_labels = np.argmax(predictions, axis=1)
predicted_letters = [label_to_letter(l) for l in predicted_labels]

true_labels = np.argmax(y_subset, axis=1) 
true_letters = [label_to_letter(l) for l in true_labels]
print("Predicted letters:", predicted_letters)
print("True letters:     ", true_letters)

plt.figure(figsize=(12,4))

for i in range(num_images):
    plt.subplot(4, 5, i+1)
    plt.imshow(X_subset[i].reshape(28,28), cmap='gray')
    plt.title(f"Pred: {predicted_letters[i]}\nTrue: {true_letters[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()