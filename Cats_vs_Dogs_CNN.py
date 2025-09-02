import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from PIL import Image
from PIL import UnidentifiedImageError

# Path to images
directory = r"D:\Escritorio\PyCharm files\Machine Learning\Cats vs Dogs Classification"
 
# Check if every image from the dataset is valid. Just has to be executed once
'''
def scan_corrupted_images(directory):
    corrupted_dir = os.path.join(directory, 'corrupted')
    os.makedirs(corrupted_dir, exist_ok=True)
    
    for root, _, files in os.walk(directory):
        # Skip processing the 'corrupted' directory
        if root == corrupted_dir:
            continue
            
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Verify image integrity
                with Image.open(file_path) as img:
                    img.verify()
                # Check if fully readable
                with Image.open(file_path) as img:
                    img.convert('RGB')
            except (UnidentifiedImageError, OSError, ValueError) as e:
                print(f"Corrupted: {file_path} - {str(e)}")
                # Move corrupted file
                dest = os.path.join(corrupted_dir, os.path.relpath(file_path, directory))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.move(file_path, dest)

scan_corrupted_images(directory)
'''

image_size = 100
batch_size = 64

# Data generators
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, # The preprocessing function should be replaced by "rescale=1/255"->[0,1] but MobileNetV2 was trained on [-1,1]
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,
                                   validation_split=0.2)

train_dataset = train_datagen.flow_from_directory(
    directory,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode="binary", # Can also be "categorical" for various classes
    subset="training"
)

validation_dataset = train_datagen.flow_from_directory(
    directory,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode="binary",
    subset="validation"
)

test_dataset = train_datagen.flow_from_directory(
    directory,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode="binary",
    shuffle=False,
    subset="validation"
)


# Function to plot a subset of the images
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5)
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

imgs, labels = train_dataset[1]
plotImages(imgs)
print(labels[:5])

# We structure our custom sequential model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Rescaling, Dense, Flatten, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Dropout

# Initialize a Sequential network
deep_model = Sequential()

# Start with the Input layer, specifying image dimensions: image_size x image_size x 3 color channels
deep_model.add(Input([image_size, image_size, 3]))

deep_model.add(Convolution2D(32, 3, activation='relu'))
deep_model.add(MaxPooling2D(2))
deep_model.add(Convolution2D(64, 3, activation='relu'))
deep_model.add(MaxPooling2D(2))
deep_model.add(Convolution2D(128, 3, activation='relu'))
deep_model.add(MaxPooling2D(2))
deep_model.add(Flatten())
deep_model.add(Dense(64, activation='relu'))
deep_model.add(Dropout(0.5))
deep_model.add(Dense(1, activation='sigmoid'))

# Compile the neural network
deep_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
deep_model.summary()# Summary of the architecture we just built

# Train the network for 10 epochs
deep_model.fit(train_dataset, epochs=10)

# Compute accuracy over the test set
loss, acc = deep_model.evaluate(test_dataset)
print(f"Loss {loss:.3}, accuracy {acc:.1%}")


# Now we would like to test how well a pre-trained model works
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Lambda

mobilenet_model = MobileNetV2(include_top=False, input_shape=(image_size, image_size, 3))
mobilenet_model.trainable = False

MobileNet_model = Sequential()
MobileNet_model.add(Input([image_size, image_size, 3]))
MobileNet_model.add(mobilenet_model)
MobileNet_model.add(Flatten())
MobileNet_model.add(Dense(64, activation='relu'))
MobileNet_model.add(Dropout(0.5))
MobileNet_model.add(Dense(1, activation='sigmoid'))

MobileNet_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])
MobileNet_model.summary()
MobileNet_model.fit(train_dataset, epochs=5)

# Compute accuracy over the test set
loss, acc = MobileNet_model.evaluate(test_dataset)
print(f"Loss {loss:.3}, accuracy {acc:.1%}")
