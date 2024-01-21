# -*- coding: utf-8 -*-


import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet18
import matplotlib.pyplot as plt
from google.colab import drive

# Mount Google Drive

drive.mount('/content/drive')


# Set the path to your dataset
dataset_path = '/content/drive/MyDrive/NNDL/MRIdataset'

# Initialize empty lists to store file paths and corresponding labels
file_paths = []
labels = []

# Iterate through each set (training and testing)
for set_folder in ['Training', 'Testing']:
    set_path = os.path.join(dataset_path, set_folder)

    # Iterate through each type of image
    for type_folder in os.listdir(set_path):
        type_path = os.path.join(set_path, type_folder)

        # Check if it's a directory
        if os.path.isdir(type_path):
            # Iterate through each image in the type folder
            for filename in os.listdir(type_path):
                # Construct the file path
                file_path = os.path.join(type_path, filename)

                # Append the file path and corresponding label to the lists
                file_paths.append(file_path)
                labels.append(type_folder)

# Create a DataFrame
df = pd.DataFrame({'File_Path': file_paths, 'Label': labels})

# Display the DataFrame
print(df.head())
df.shape

################## Convolutional Neural Networks(CNN ) Model #################


# Split the DataFrame into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)

# Image dimensions and channels
img_height, img_width = 150, 150
img_channels = 3  # Assuming RGB images

# Number of classes
num_classes = len(df['Label'].unique())

# Data Augmentation for Training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling for Testing
test_datagen = ImageDataGenerator(rescale=1./255)

# Training Data Generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='File_Path',
    y_col='Label',
    target_size=(img_height, img_width),
    class_mode='categorical',
    batch_size=512
)

# Testing Data Generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='File_Path',
    y_col='Label',
    target_size=(img_height, img_width),
    class_mode='categorical',
    batch_size=64
)

# CNN Model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, img_channels)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))



# Model Compilation
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model Training
history = model.fit(train_generator, epochs=30, validation_data=test_generator)

# Model Evaluation
loss, accuracy = model.evaluate(test_generator)


#### plotting the CNN training accuracy ####



# Plot training history
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.show()


###################################     VGG-16 Model    #####################################



# df is defined in the code at starting

# Split the DataFrame into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)

# Image dimensions
img_height, img_width = 224, 224  # VGG16 input size

# Batch size
batch_size = 512

# Number of classes
num_classes = len(df['Label'].unique())

# Data Augmentation for Training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling for Testing
test_datagen = ImageDataGenerator(rescale=1./255)

# Training Data Generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='File_Path',
    y_col='Label',
    target_size=(img_height, img_width),
    class_mode='categorical',
    batch_size=batch_size
)

# Testing Data Generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='File_Path',
    y_col='Label',
    target_size=(img_height, img_width),
    class_mode='categorical',
    batch_size=batch_size
)

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the pre-trained model
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

# Model Compilation
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model Training
history = model.fit(train_generator, epochs=30, validation_data=test_generator)

# Model Evaluation
loss, accuracy = model.evaluate(test_generator)

         ######### plotting the VGG 16 training accuracy ############


# Plot training history
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.show()


###########################################   ResNet-18 Model     #####################################



# df is defined in the code at start

# Split the DataFrame into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)

# Image dimensions
img_height, img_width = 224, 224  # ResNet18 input size

# Batch size
batch_size = 512

# Number of classes
num_classes = len(df['Label'].unique())

# Data Augmentation for Training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling for Testing
test_datagen = ImageDataGenerator(rescale=1./255)

# Training Data Generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='File_Path',
    y_col='Label',
    target_size=(img_height, img_width),
    class_mode='categorical',
    batch_size=batch_size
)

# Testing Data Generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='File_Path',
    y_col='Label',
    target_size=(img_height, img_width),
    class_mode='categorical',
    batch_size=batch_size
)

# Load pre-trained ResNet18 model
base_model = ResNet18(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of the pre-trained model
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

# Model Compilation
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model Training
history = model.fit(train_generator, epochs=30, validation_data=test_generator)

# Model Evaluation
loss, accuracy = model.evaluate(test_generator)

        ###########    plotting ResNet18 Model ############

# Plot training history
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.show()