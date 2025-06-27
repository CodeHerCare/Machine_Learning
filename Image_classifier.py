import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tqdm import tqdm
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def clean_single_image(image_filename, size=(224, 224), convert_to_gray=False):
    try:
        img = Image.open(image_filename)
        img.verify()

        img = Image.open(image_filename)

        if convert_to_gray:
            img = img.convert("L")

        img = img.resize(size)   

        np_img = np.array(img) / 255.0
        img = Image.fromarray((np_img * 255).astype('uint8'))

        return np_img  # return numpy array for model input

    except Exception as e:
        print(f"Image processing failed: {e}")
        return None

def load_dataset(csv_path, img_size=(224, 224), convert_to_gray=False):
    df = pd.read_csv(csv_path)
    images = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img = clean_single_image(row['image_filename'], size=img_size, convert_to_gray=convert_to_gray)
        if img is not None:
            images.append(img)
            labels.append(row['label'])

    images = np.array(images)
    # If grayscale, add channel dimension
    if convert_to_gray:
        images = images.reshape(-1, img_size[0], img_size[1], 1)
    else:
        images = images.reshape(-1, img_size[0], img_size[1], 3)  # Assuming RGB

    return images, np.array(labels)

def image_classifier(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def run_pipeline(csv_path):
    X, y = load_dataset(csv_path)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = keras.utils.to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    model = image_classifier(input_shape=X_train.shape[1:], num_classes=y_cat.shape[1])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")

    model.save("cervical_cancer_cnn_model.h5")

if __name__ == "__main__":
    run_pipeline(r"C:\Users\USER\Desktop\CHC\Cervical_cancer_Data\classifications.csv")
