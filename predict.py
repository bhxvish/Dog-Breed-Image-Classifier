import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import joblib
model = tf.keras.models.load_model("dog_breed_model.h5")
label_encoder = joblib.load("label_encoder.pkl")

def predict_dog_breed(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    breed = label_encoder.inverse_transform([np.argmax(prediction)])
    return breed[0]