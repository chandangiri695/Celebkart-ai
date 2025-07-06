import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from gtts import gTTS
import os
from tempfile import NamedTemporaryFile
from deep_translator import GoogleTranslator

# Load BLIP model
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Title
st.title("👗 Celebkart AI - Outfit Identifier with Voice & Translation")

# File uploader
uploaded_file = st.file_uploader("Upload a celeb or fashion photo 👇", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process image
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)

    st.markdown("### ✅ AI Description:")
    st.success(description)

    # 🔁 Translation
    translated = GoogleTranslator(source='auto', target='hi').translate(description)
    st.markdown("### 🌐 Hindi Translation:")
    st.info(translated)

    # 🔊 Voice output
    tts = gTTS(text=description, lang='en')
    with NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")

    # 🛍️ Dummy Product Suggestions
    st.markdown("### 🛒 Suggested Products:")
    st.write("- Green cotton shirt - ₹999 on Myntra")
    st.write("- Beige trousers - ₹1,499 on Amazon")
    st.write("- Brown formal shoes - ₹1,999 on Flipkart")

    st.markdown("**(These are sample links, affiliate integration coming soon...)**")
