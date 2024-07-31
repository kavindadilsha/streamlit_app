import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# Load the pre-trained model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# Function to decode predictions
def decode_predictions(predictions):
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    return decoded

# Set up the Streamlit app
st.title("Image Classification App")
st.write("Upload an image to classify")

# Upload image widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    
    # Add a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Update progress
    progress_bar.progress(50)
    status_text.text("Loading image...")
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Update progress
    progress_bar.progress(75)
    status_text.text("Making predictions...")
    
    # Make predictions
    predictions = model.predict(processed_image)
    decoded_predictions = decode_predictions(predictions)
    
    # Update progress
    progress_bar.progress(100)
    status_text.text("Done!")
    
    # Display predictions
    st.write("Predictions:")
    for pred in decoded_predictions:
        st.write(f"{pred[1]}: {pred[2]*100:.2f}%")

# Sidebar for additional inputs
with st.sidebar:
    st.title("Additional Settings")
    st.write("You can add more settings here.")
    
# Example graph
st.write("Example Graph:")
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)
st.line_chart(chart_data)
