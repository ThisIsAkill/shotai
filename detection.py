import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def load_data(data_dir):
    images = []
    shot_labels = []
    angle_labels = []
    for shot_type in os.listdir(data_dir):
        shot_dir = os.path.join(data_dir, shot_type)
        if not os.path.isdir(shot_dir):
            continue
        for angle_type in os.listdir(shot_dir):
            angle_dir = os.path.join(shot_dir, angle_type)
            if not os.path.isdir(angle_dir):
                continue
            for image_file in os.listdir(angle_dir):
                image_path = os.path.join(angle_dir, image_file)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (224, 224))
                    images.append(image)
                    shot_labels.append(shot_type)
                    angle_labels.append(angle_type)
    if len(images) == 0:
        raise ValueError("No images found in dataset. Please check the dataset directory and structure.")
    return np.array(images), np.array(shot_labels), np.array(angle_labels)

# Update the data directory path if you're on Windows
data_dir = 'G:\\GitHub\\shotsai\\datasets'

# Load and preprocess data
print("Loading data...")
try:
    images, shot_labels, angle_labels = load_data(data_dir)
except ValueError as e:
    print(e)
    exit()

# Normalize images
print("Normalizing images...")
images = images / 255.0

# Encode labels
print("Encoding labels...")
shot_label_encoder = LabelEncoder()
angle_label_encoder = LabelEncoder()
shot_labels = shot_label_encoder.fit_transform(shot_labels)
angle_labels = angle_label_encoder.fit_transform(angle_labels)

# Split data
print("Splitting data into training and testing sets...")
X_train, X_test, y_shot_train, y_shot_test, y_angle_train, y_angle_test = train_test_split(
    images, shot_labels, angle_labels, test_size=0.2, random_state=42
)

# Data augmentation
print("Creating data generator...")
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Model definition
print("Building model...")
input_layer = Input(shape=(224, 224, 3))

x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layers
shot_output = Dense(len(np.unique(shot_labels)), activation='softmax', name='shot_output')(x)
angle_output = Dense(len(np.unique(angle_labels)), activation='softmax', name='angle_output')(x)

model = Model(inputs=input_layer, outputs=[shot_output, angle_output])

# Compile the model
model.compile(optimizer='adam', 
              loss={'shot_output': 'sparse_categorical_crossentropy', 'angle_output': 'sparse_categorical_crossentropy'}, 
              metrics=['accuracy'])

# Summary of the model
model.summary()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
print("Training model...")
history = model.fit(
    datagen.flow(X_train, {'shot_output': y_shot_train, 'angle_output': y_angle_train}, batch_size=32),
    epochs=50,
    validation_data=(X_test, {'shot_output': y_shot_test, 'angle_output': y_angle_test}),
    callbacks=[early_stopping, model_checkpoint],
    verbose=2  # Set verbosity level to 2 for detailed output
)

# Evaluate the model
print("Evaluating model...")
test_loss, shot_test_loss, angle_test_loss, shot_test_acc, angle_test_acc = model.evaluate(X_test, 
    {'shot_output': y_shot_test, 'angle_output': y_angle_test})
print(f'Test accuracy for shot types: {shot_test_acc}')
print(f'Test accuracy for camera angles: {angle_test_acc}')

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['shot_output_accuracy'], label='Shot Type Accuracy')
    plt.plot(history.history['val_shot_output_accuracy'], label = 'Val Shot Type Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['angle_output_accuracy'], label='Camera Angle Accuracy')
    plt.plot(history.history['val_angle_output_accuracy'], label = 'Val Camera Angle Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    plt.show()

print("Plotting training history...")
plot_history(history)

# Classification report and confusion matrix
def evaluate_model(X_test, y_shot_test, y_angle_test, model):
    y_shot_pred, y_angle_pred = model.predict(X_test)
    y_shot_pred = np.argmax(y_shot_pred, axis=1)
    y_angle_pred = np.argmax(y_angle_pred, axis=1)

    print("Shot Type Classification Report:")
    print(classification_report(y_shot_test, y_shot_pred, target_names=shot_label_encoder.classes_))

    print("Camera Angle Classification Report:")
    print(classification_report(y_angle_test, y_angle_pred, target_names=angle_label_encoder.classes_))

    print("Shot Type Confusion Matrix:")
    cm_shot = confusion_matrix(y_shot_test, y_shot_pred)
    sns.heatmap(cm_shot, annot=True, fmt='d', xticklabels=shot_label_encoder.classes_, yticklabels=shot_label_encoder.classes_)
    plt.show()

    print("Camera Angle Confusion Matrix:")
    cm_angle = confusion_matrix(y_angle_test, y_angle_pred)
    sns.heatmap(cm_angle, annot=True, fmt='d', xticklabels=angle_label_encoder.classes_, yticklabels=angle_label_encoder.classes_)
    plt.show()

print("Evaluating model performance...")
evaluate_model(X_test, y_shot_test, y_angle_test, model)

# Function for prediction
def predict_shot_and_angle(image_path, model, shot_label_encoder, angle_label_encoder):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    
    shot_prediction, angle_prediction = model.predict(image)
    predicted_shot_label = shot_label_encoder.inverse_transform([np.argmax(shot_prediction)])
    predicted_angle_label = angle_label_encoder.inverse_transform([np.argmax(angle_prediction)])
    
    return predicted_shot_label[0], predicted_angle_label[0]

# Example usage
image_path = r'C:\Users\thisisakill\Downloads\tr-2022-283-20427.jpg'
print("Predicting shot type and camera angle for an example image...")
shot_type, angle_type = predict_shot_and_angle(image_path, model, shot_label_encoder, angle_label_encoder)
print(f'The predicted shot type is: {shot_type}')
print(f'The predicted camera angle is: {angle_type}')
