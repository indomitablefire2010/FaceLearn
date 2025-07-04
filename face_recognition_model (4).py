import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to your 'faces' folder where your images are stored
train_dir = r"C:\Users\asus\Desktop\data\Faces\Faces"  # Replace this with your actual folder path

# Set up ImageDataGenerator to load images and rescale them
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Rescale images to [0,1]
    validation_split=0.2      # 20% of the data for validation
)

# Set up the data generator for training images
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to 224x224 (standard input size for VGG16)
    batch_size=32,           # Batch size (you can adjust this depending on your system)
    class_mode='categorical', # Since you're classifying different persons
    subset='training'        # Use the training subset
)

# Set up the data generator for validation images
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to 224x224 (standard input size for VGG16)
    batch_size=32,           # Batch size (you can adjust this depending on your system)
    class_mode='categorical', # Since you're classifying different persons
    subset='validation'      # Use the validation subset
)

# Check if data is loaded properly
print(f"Classes: {train_generator.class_indices}")
