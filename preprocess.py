import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset folder path
dataset_path = "dataset/"

# Image generator (resize + normalize + split)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

print("Preprocessing Done ")