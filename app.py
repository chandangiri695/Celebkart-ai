import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

st.set_page_config(page_title="Celebkart AI", layout="centered")
st.title("👗 Celebkart AI - Celebrity Outfit Identifier")
st.caption("Upload any celebrity photo. AI will describe their outfit automatically.")

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

processor, model = load_model()

uploaded_file = st.file_uploader("📤 Upload a celebrity outfit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("AI is analyzing the outfit..."):
        inputs = processor(images=image, text="Describe the outfit the person is wearing", return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

    st.success("✅ AI Description:")
    st.markdown(f"**{caption}**")
    st.info("You can copy this text and search it on Amazon, Myntra, or Flipkart manually.")
