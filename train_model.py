import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import os
import kagglehub

img_size = 150

# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)

train_dir = 'dataset/train'

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(img_size, img_size),
    batch_size=32
)
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_size, img_size, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])  
model.compile(optimizer='adam',
              loss='binary_crossentropy',
                metrics=['accuracy'])
model.fit(train_ds, epochs=5)   
model.save('model.h5')
