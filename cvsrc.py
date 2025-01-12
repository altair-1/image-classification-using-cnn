# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


# Step 1: Load and Preprocess the CIFAR-10 Dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize the image data (scale pixel values to between 0 and 1)
train_images, test_images = train_images / 255.0, test_images / 255.0

# Convert labels to categorical (one-hot encoding)
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# Step 2: Define the CNN Model with Batch Normalization and Dropout
model = models.Sequential()

# Input layer
model.add(layers.Input(shape=(32, 32, 3)))

# First Conv2D Layer with Batch Normalization and Dropout
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))  # Dropout to prevent overfitting

# Second Conv2D Layer with Batch Normalization and Dropout
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))  # Dropout to prevent overfitting

# Third Conv2D Layer with Batch Normalization and Dropout
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.25))  # Dropout to prevent overfitting

# Flattening the output
model.add(layers.Flatten())

# Fully Connected Layer with Batch Normalization and Dropout
model.add(layers.Dense(128, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))  # Larger dropout for fully connected layer

# Output Layer (Softmax for classification)
model.add(layers.Dense(10, activation='softmax'))

# Model summary
model.summary()

# Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 3: Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2
)

# Fit the data generator to the training data
datagen.fit(train_images)

# Step 4: Train the Model Using Augmented Data
history_augmented = model.fit(
    datagen.flow(train_images, train_labels, batch_size=64),
    validation_data=(test_images, test_labels),
    epochs=10
)
# optional but better to save the model
model.save('cifar10_model.h5')

# Step 5: Evaluate the Model on Test Data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# Step 6: Visualize Training and Validation Accuracy and Loss
epochs = range(1, 11)

plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs, history_augmented.history['accuracy'], 'b', label='Training accuracy')
plt.plot(epochs, history_augmented.history['val_accuracy'], 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs, history_augmented.history['loss'], 'b', label='Training loss')
plt.plot(epochs, history_augmented.history['val_loss'], 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# Step 7: Generate Confusion Matrix
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_labels, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    # Load image with target size matching the input shape of the model
    img = image.load_img(img_path, target_size=(32, 32))
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Reshape the image array (Add batch dimension, i.e., (1, 32, 32, 3))
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image data (scale pixel values to between 0 and 1)
    img_array = img_array / 255.0
    return img_array

# Function to classify the image
def classify_image(model, img_path, class_names):
    # Preprocess the image
    img_array = load_and_preprocess_image(img_path)
    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    
    # Show the image
    img = image.load_img(img_path, target_size=(32, 32))
    plt.imshow(img)
    plt.title(f"Predicted: {class_names[predicted_class]}, Confidence: {confidence:.4f}")
    plt.show()

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Example usage (replace 'your_image_path_here' with the path of the image you want to classify)
image_path = 'car1.jpeg'  # Modify this with the actual image path
classify_image(model, image_path, class_names)