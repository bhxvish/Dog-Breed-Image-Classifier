import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import joblib
from tensorflow.keras.layers import Dropout

labels_df = pd.read_csv("labels.csv")
labels_df["id"] = labels_df["id"].astype(str) + ".jpg"
le = LabelEncoder()
labels_df["breed_encoded"] = le.fit_transform(labels_df["breed"])
joblib.dump(le, "label_encoder.pkl")
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory= "train/",
    x_col = "id",
    y_col = "breed",
    subset = "training",
    target_size = (224, 224),
    class_mode= "categorical",
    batch_size = 32
)

val_gen = datagen.flow_from_dataframe(
    dataframe=labels_df,
    directory="train/",
    x_col= "id",
    y_col = "breed",
    target_size = (224,224),
    class_mode="categorical",
    batch_size = 32
)

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = Sequential([base_model,
                    GlobalAveragePooling2D(),
                    Dropout(0.3),
                    Dense(120, activation = "softmax")
                    ])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
checkpoint = ModelCheckpoint("dog_breed_model.h5", save_best_only=True)
model.fit(train_gen, epochs=5, validation_data = val_gen, callbacks=[checkpoint])



