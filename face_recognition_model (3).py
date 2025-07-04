from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = load_model('face_recognition_model.keras')  # or 'face_recognition_model.h5' if you're using that format

# Define paths
train_dir = r'C:/Users/asus/Desktop/data/Faces/Faces'

# Use ImageDataGenerator with validation_split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load training and validation data with the correct target size
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Update this line to match the input size of the model
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Update this line to match the input size of the model
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(validation_generator)
print("Validation accuracy:", val_accuracy)
