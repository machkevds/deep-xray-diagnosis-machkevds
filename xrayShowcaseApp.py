import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import tempfile
import os

# setup
# image size and class names
IMG_SIZE = (150, 150)
CLASS_NAMES = ["Normal", "Pneumonia"]
LAST_CONV_LAYER_NAME = "Conv_1"
THRESHOLD = 0.4  # arbitrary threshold

# load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_mobilenetv2_kvpch10.keras")  # Change to your model path

model = load_model()

# helper functions

def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array, np.array(img)



def predict(img_array):
    prob = model.predict(img_array)[0][0]
    label = CLASS_NAMES[1] if prob > THRESHOLD else CLASS_NAMES[0]
    score = prob if prob > THRESHOLD else 1 - prob
    return label, score, prob



def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model(inputs=model.inputs, outputs=[
        model.get_layer(last_conv_layer_name).output,
        model.output
    ])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()




def overlay_gradcam(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap_color * 0.4 + original_img
    superimposed_img = np.clip(superimposed_img / 255.0, 0, 1)
    return superimposed_img



# streamlit UI
st.title(" :hospital: Pneumonia Detection from Chest X-ray :stethoscope:")
st.write("Upload a chest X-ray image below to get a prediction.	:point_down:")

uploaded_file = st.file_uploader("Choose a chest X-ray image :open_file_folder:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # preprocess
    img_array, original_img = preprocess_image(uploaded_file)
    
    # predict
    label, score, prob = predict(img_array)

    # display prediction
    st.subheader("Prediction:")
    st.write(f"**{label}** ({score*100:.2f}% confidence)")

    col1, col2 = st.columns(2)

    with col1:
        # display test image
        st.markdown("#### Uploaded X-ray")
        st.image(original_img, width=300) 

    with col2:
        # Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)
        gradcam_img = overlay_gradcam(original_img, heatmap)
        st.markdown('#### Grad-CAM Overlay')
        st.image(gradcam_img, width=300)