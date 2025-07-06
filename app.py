import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Cache the model so it doesn't reload every time
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

st.title("👕 Celebkart AI")
st.markdown("Upload a celebrity image to get AI-generated outfit description.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing outfit..."):
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs)
        description = processor.decode(output[0], skip_special_tokens=True)

    st.success("✅ AI Description:")
    st.markdown(f"**{description}**")
