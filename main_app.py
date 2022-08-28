import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Loading models

model = load_model('dog_breed.h5')

# Name of classes
CLASS_NAME = ['Scottish Deerhound', 'Malteste Dog', 'Bernese Mountain Dog']

# Setting title for App
st.title("Dog Breed Prediction")
st.markdown("Upload an Image of the Dog")

# Uploading the dog image
dog_image = st.file_uploader("Choose an Image to Upload...", type="png")
submit = st.button("Prediction")
# on prediction click

if submit:
    if dog_image is not None:
        # Convert the dog image to a opencv image
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the dog image
        st.image(opencv_image, channels="BGR")
        # Reszing the dog image
        opencv_image = cv2.resize(opencv_image, (224, 224))
        # Convert the dog image to a 4 dimension
        opencv_image.shape = (1, 224, 224, 3)


        # Predicting the dog
        Y_pred = model.predict(opencv_image)

        st.title(str("The dog Breed is " + CLASS_NAME[np.argmax(Y_pred)]))


        # Displaying the prediction
