# Celebkart v2.0 - Advanced Outfit Identifier with Affiliate Integration

# Celebkart v2.0 - Advanced Outfit Identifier with Affiliate Integration

import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
from gtts import gTTS
import os
from tempfile import NamedTemporaryFile
Load BLIP model

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base") model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

st.set_page_config(page_title="Celebkart v2.0 - Outfit AI", layout="centered") st.title("👕 Celebkart v2.0 - Outfit Identifier + Shop Links")

uploaded_file = st.file_uploader("Upload an image of a person", type=["jpg", "jpeg", "png"]) language = st.selectbox("Choose description language", ["English", "Hindi"])

Helper function to get outfit description

def describe_outfit(image): inputs = processor(images=image, return_tensors="pt") out = model.generate(**inputs) description = processor.decode(out[0], skip_special_tokens=True) return description

Simulated product suggestions based on keywords

def get_mock_products(description): mock = { "green shirt": ("Olive Green Shirt", 599, "https://www.amazon.in/dp/B0BMock1"), "beige pants": ("Beige Chinos", 999, "https://www.flipkart.com/p/BeigeMock2"), "blue shoes": ("Blue Slip-on Shoes", 799, "https://myntra.com/p/BlueMock3"), "suit": ("Formal Navy Blue Suit", 2499, "https://amazon.in/p/SuitMock4") } results = [] for k, v in mock.items(): if k in description.lower(): results.append(v) return results

Translate to Hindi (basic)

def translate_to_hindi(text): try: from deep_translator import GoogleTranslator return GoogleTranslator(source='auto', target='hi').translate(text) except: return text

Text-to-speech audio

@st.cache_resource def generate_tts(text, lang): with NamedTemporaryFile(delete=False, suffix=".mp3") as fp: tts = gTTS(text=text, lang='hi' if lang == 'Hindi' else 'en') tts.save(fp.name) return fp.name

if uploaded_file: image = Image.open(uploaded_file).convert('RGB') st.image(image, caption="Uploaded Image", use_column_width=True) st.info("🔍 Detecting outfit...") description = describe_outfit(image)

if language == "Hindi":
    translated = translate_to_hindi(description)
    st.success(f"✅ AI Description (Hindi): {translated}")
else:
    st.success(f"✅ AI Description: {description}")

# Product suggestions
st.markdown("### 🛍️ Suggested Items:")
for title, price, link in get_mock_products(description):
    st.markdown(f"- **{title}** – ₹{price} – [Buy Now]({link})")

# Voice
st.markdown("### 🔊 Voice Description:")
audio_path = generate_tts(description if language == "English" else translated, language)
st.audio(audio_path)

# Share option (for WhatsApp/web)
st.markdown("📤 Share your AI result with friends!")
st.text(f"Outfit AI: {description}")

