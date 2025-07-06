import streamlit as st
from PIL import Image
import requests
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Clear cache to avoid memory issues
st.cache_resource.clear()

# Load BLIP Large model
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

processor, model = load_model()

st.title("👕 Celebkart AI – Outfit Identifier")

uploaded_file = st.file_uploader("Upload an image of the outfit", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.markdown("✅ **AI Description:**")

    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    st.write(description)
