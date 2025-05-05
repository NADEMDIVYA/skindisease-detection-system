import streamlit as st
import numpy as np
import base64
import openvino as ov
from PIL import Image

# ✅ Function to add background image in Streamlit
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        base64_img = base64.b64encode(f.read()).decode()
    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# ✅ Set background image
add_bg_from_local("background.jpg")  # Ensure this image is in the same directory

# Load the OpenVINO model
core = ov.Core()
model_ir = core.read_model(model="openvino_model/best_skin_model.xml")
compiled_model = core.compile_model(model_ir, "CPU")

# Get input and output layer info
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Class labels
class_names = [
    'Atopic Dermatitis', 'Basal Cell Carcinoma', 'Benign Keratosis-like Lesions',
    'Eczema', 'Melanocytic Nevi', 'Melanoma', 'Psoriasis (Lichen Planus)',
    'Seborrheic Keratoses and other benign tumors', 'Tinea (Ringworm)',
    'Unknown',  # Non-skin class
    'Warts, Molluscum, and Viral Infections'
]

# Medicine suggestions
medicine_dict = {
    'Eczema': 'Hydrocortisone Cream',
    'Melanoma': 'Consult Oncologist Immediately',
    'Atopic Dermatitis': 'Cetirizine + Moisturizers',
    'Basal Cell Carcinoma': 'Surgical Removal, consult Dermatologist',
    'Melanocytic Nevi': 'Observation or Surgical Removal',
    'Benign Keratosis-like Lesions': 'Topical Retinoids',
    'Psoriasis (Lichen Planus)': 'Topical Steroids + Vitamin D Analogues',
    'Seborrheic Keratoses and other benign tumors': 'Cryotherapy',
    'Tinea (Ringworm)': 'Antifungal Cream - Clotrimazole',
    'Warts, Molluscum, and Viral Infections': 'Salicylic Acid or Cryotherapy',
}

# Streamlit UI
st.title("🩺 Skin Disease Detection System")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Predict
    result = compiled_model(img_array)[output_layer]
    predicted_idx = np.argmax(result)
    confidence = float(np.max(result))
    predicted_class = class_names[predicted_idx]

    # Block if overconfident
    if confidence >= 0.99:
        st.warning("⚠️ The uploaded image appears to be invalid or not a skin disease.")
        st.caption(f"Prediction blocked (High confidence: {confidence:.2f})")
    else:
        st.success(f"✅ Predicted Disease: **{predicted_class}**")
        if predicted_class != "Unknown":
            st.info(f"💊 Suggested Medicine: **{medicine_dict.get(predicted_class, 'Consult Doctor')}**")
        else:
            st.warning("⚠️ The uploaded image may not clearly represent a known skin disease.")
        st.caption(f"Confidence: {confidence:.2f}")
