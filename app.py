import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from PIL import Image
import numpy as np

st.title("Skin Cancer / Image Detection App")
st.write("Upload an image to classify using ResNet-50")

@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet')

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=3)[0]
    
    st.subheader("Predictions:")
    for (_, label, prob) in decoded_preds:
        st.write(f"**{label}**: {prob*100:.2f}%")
