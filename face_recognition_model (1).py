from tensorflow.keras.models import load_model

# Load the previously saved model (replace with your actual file path)
model = load_model('face_recognition_model.h5')

# Save the model in Keras format
model.save('face_recognition_model.keras')
