import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2
import numpy as np
import pickle
from numpy.linalg import norm
import uuid
import shutil

# Function to load pickle file
def load_pickle_file(filepath):
    try:
        with open(filepath, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading {filepath}: {e}")
        return None

# Load the preprocessed features and filenames
resnet50_features = load_pickle_file('pickle files/resnet50_features.pkl')
color_features = load_pickle_file('pickle files/color_features.pkl')
filenames = load_pickle_file('pickle files/filenames.pkl')

if resnet50_features is not None and color_features is not None:
    resnet50_features = np.array(resnet50_features)
    color_features = np.array(color_features)
    # Combine both features
    combined_features = np.hstack((resnet50_features, color_features))
else:
    combined_features = None

# Define the model for feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def save_uploaded_file(uploaded_file):
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        base_filename = os.path.basename(uploaded_file.name)
        dest_path = os.path.join('uploads', base_filename)
        with open(dest_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return dest_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def save_recommended_images(images):
    folder_name = f"recommendations_{uuid.uuid4().hex}"
    folder_path = os.path.join('uploads', folder_name)
    os.makedirs(folder_path, exist_ok=True)
    for img_path in images:
        img_name = os.path.basename(img_path)
        dest_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        cv2.imwrite(dest_path, img)
    return folder_path

def extract_resnet50_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def extract_color_histogram(image_path, bins=(8, 8, 8)):
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def feature_extraction(img_path, model):
    resnet50_feature = extract_resnet50_features(img_path, model)
    color_feature = extract_color_histogram(img_path)
    combined_feature = np.hstack((resnet50_feature, color_feature))
    return combined_feature

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Load recent recommendations from file
def load_recent_recommendations():
    try:
        with open('recent_recommendations.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Error loading recent recommendations: {e}")
        return []

# Save recent recommendations to file
def save_recent_recommendations(recommendations):
    try:
        with open('recent_recommendations.pkl', 'wb') as file:
            pickle.dump(recommendations, file)
    except Exception as e:
        st.error(f"Error saving recent recommendations: {e}")

# Initialize or load recent recommendations
if 'recent_recommendations' not in st.session_state:
    st.session_state.recent_recommendations = load_recent_recommendations()

# Streamlit app
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["About", "Recommendations", "Recent Recommendations", "Contact"])

if options == "About":
    st.title("About")
    st.write("## Overview")
    st.write("This application is an AI fashion recommendation system that suggests similar images based on a given input image.")
    
    st.write("## How It Works")
    
    st.write("### 1. Preprocessing and Feature Extraction")
    st.write("""
    The model leverages a pre-trained ResNet50 neural network to extract deep features from images. ResNet50 is a powerful convolutional neural network trained on ImageNet. In this system:
    - The top layer of ResNet50 is removed, and a `GlobalMaxPooling2D` layer is added to get feature vectors.
    - Each image is resized to 224x224 pixels, preprocessed, and then passed through the modified ResNet50 to obtain a feature vector.
    - In addition to ResNet50 features, a color histogram is computed for each image, capturing color distribution in the HSV color space.
    - Both feature types are normalized and combined into a single feature vector for each image.
    """)
    
    st.write("### 2. Model Training")
    st.write("""
    The combined features are saved into pickle files for quick access. A nearest neighbors model (`sklearn.neighbors.NearestNeighbors`) is used to find the most similar images based on the Euclidean distance between feature vectors.
    """)
    
    st.write("### 3. Making Recommendations")
    st.write("""
    When a user uploads an image:
    - The system extracts ResNet50 and color histogram features from the uploaded image.
    - These features are combined and compared with the precomputed features using the nearest neighbors model.
    - The system retrieves and displays the most similar images.
    """)
    
elif options == "Recommendations":
    st.title("AI Fashion Recommendations")
    st.subheader("Get Recommendations By Uploading an Image")
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        saved_file_path = save_uploaded_file(uploaded_file)
        if saved_file_path:
            st.write("Uploaded Image:")
            st.image(saved_file_path, width=300)
            
            if combined_features is not None:
                features = feature_extraction(saved_file_path, model)
                indices = recommend(features, combined_features)
                
                st.write("Recommended Images:")
                recommended_images = [filenames[idx] for idx in indices[0]]
                cols = st.columns(len(recommended_images))  # Create columns for each image
                for col, img_path in zip(cols, recommended_images):
                    col.image(img_path, use_column_width=True)  # Display each image in a separate column
                
                # Save the recommended images in a unique folder
                folder_path = save_recommended_images(recommended_images)
                st.session_state.recent_recommendations.append((saved_file_path, folder_path))
                save_recent_recommendations(st.session_state.recent_recommendations)
            else:
                st.error("Features are not loaded correctly. Please check the pickle files.")
        
elif options == "Recent Recommendations":
    st.title("Recent Recommendations")

    if st.session_state.recent_recommendations:
        if st.button("Delete All Recommendations"):
            for uploaded_img, folder_path in st.session_state.recent_recommendations:
                shutil.rmtree(folder_path)  # Remove the folder with recommended images
                os.remove(uploaded_img)  # Remove the uploaded image
            st.session_state.recent_recommendations.clear()
            save_recent_recommendations(st.session_state.recent_recommendations)
            st.success("All recommendations deleted.")
            st.rerun()  # Rerun to refresh the display
        
        for i, (uploaded_img, folder_path) in enumerate(st.session_state.recent_recommendations):
            with st.container():
                st.write("Uploaded Image:")
                st.image(uploaded_img, width=300)
                
                st.write("Recommended Images:")
                recommended_imgs = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]
                cols = st.columns(len(recommended_imgs))  # Create columns for each image
                for j, (col, img_path) in enumerate(zip(cols, recommended_imgs)):
                    with col:
                        col.image(img_path, use_column_width=True)  # Display each image in a separate column
                        if col.button(f"Delete Image {j + 1}", key=f"{i}-{j}"):
                            os.remove(img_path)  # Remove the image from the file system
                            recommended_imgs.pop(j)  # Remove the image from the list
                            # Check if folder is empty after deletion
                            if not os.listdir(folder_path):
                                st.session_state.recent_recommendations.pop(i)
                                st.success(f"Recommendation {i + 1} deleted because all images were removed.")
                                shutil.rmtree(folder_path)  # Remove the folder if empty
                                os.remove(uploaded_img)  # Remove the uploaded image
                            save_recent_recommendations(st.session_state.recent_recommendations)
                            st.rerun()  # Rerun to refresh the display
                
                if st.button(f"Delete Recommendation {i + 1}", key=f"delete_rec_{i}"):
                    st.session_state.recent_recommendations.pop(i)
                    shutil.rmtree(folder_path)  # Remove the folder with recommended images
                    os.remove(uploaded_img)  # Remove the uploaded image
                    st.success(f"Recommendation {i + 1} deleted.")
                    save_recent_recommendations(st.session_state.recent_recommendations)
                    st.rerun()  # Rerun to refresh the display
    else:
        st.write("No recent recommendations available.")

elif options == "Contact":
    st.title("Contact")
    st.write("For any queries, please contact us at: example@example.com")
