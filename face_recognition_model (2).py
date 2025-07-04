from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# dataset path
train_dir = r'C:/Users/asus/Desktop/data/Faces/Faces'

# ImageDataGenerator to load and augment images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Split the data into 80% train and 20% validation
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Use the training subset
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Use the validation subset
)

# Load the VGG16 model (without the top layer, since we want to customize the output layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the first few layers of VGG16 and unfreeze some later layers for fine-tuning
for layer in base_model.layers[:15]:  # Freeze the first 15 layers
    layer.trainable = False
for layer in base_model.layers[15:]:  # Unfreeze the remaining layers for fine-tuning
    layer.trainable = True

# Create the model
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(31, activation='softmax'))  # 31 classes in the dataset 'Faces'

# Compile the model with a smaller learning rate to prevent overshooting
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Implement EarlyStopping to stop training if the accuracy stops improving
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

# Train the model with validation data
history = model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Save the model after training
model.save('face_recognition_model.h5')
