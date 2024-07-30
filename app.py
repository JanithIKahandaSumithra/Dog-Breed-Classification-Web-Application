import streamlit as st
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import joblib
from PIL import Image
import time
import pandas as pd
import matplotlib.pyplot as plt

# Load the model and class labels
try:
    model = load_model('dog_breed_model.h5', compile=False)  # Use compile=False if you encounter issues
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

try:
    class_labels = joblib.load('class_labels.pkl')
except Exception as e:
    st.error(f"Error loading class labels: {e}")
    st.stop()

# Streamlit app
st.title('Dog Breed Classification')

# Sidebar for the number of predictions to display
st.sidebar.title('Options')
num_predictions = st.sidebar.slider('Number of Predictions to Display', min_value=1, max_value=10, value=5)

# Upload image
st.write("Upload an image of a dog to classify:")

# Create a container for the image upload section
with st.container():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Show the image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image.', use_column_width=True)
        
        # Prepare image for model prediction
        img = img.resize((150, 150))  # Resize image to model's expected input size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Display a status message and a progress bar
        st.write("Processing the image and making prediction...")
        
        with st.spinner('Loading model and making prediction...'):
            # Simulate some delay for model loading and prediction
            time.sleep(2)  # Adjust or remove this for real processing time
            try:
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions[0])
                
                # Display result with enhanced styling
                st.markdown(f"### **Prediction:** {class_labels[predicted_class]}")
                
                # Convert probabilities to percentage
                probabilities = predictions[0] * 100
                
                # Show prediction probabilities in percentage
                st.write("### Prediction Probabilities:")
                
                # Prepare data for the table
                prob_data = {
                    'Dog Breed': class_labels,
                    'Probability (%)': [f"{prob:.2f}%" for prob in probabilities]
                }
                prob_df = pd.DataFrame(prob_data)
                
                # Display the probabilities table
                st.table(prob_df.sort_values(by='Probability (%)', ascending=False).head(num_predictions))
                
                # Display a bar chart of the prediction probabilities
                st.write("### Prediction Probabilities Chart:")
                fig, ax = plt.subplots()
                ax.bar(class_labels, probabilities)
                ax.set_xlabel('Dog Breeds')
                ax.set_ylabel('Probability (%)')
                ax.set_title('Prediction Probabilities')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
