import streamlit as st
from predict import predict_dog_breed

st.title("Dog Breed Classifier")
file = st.file_uploader("Upload a dog image", type=["jpg","jpeg","png"])
if file:
    with open("temp.jpg", "wb") as f:
        f.write(file.getbuffer())

    st.image("temp.jpg", caption="Uploaded Image", use_container_width=True)
    breed = predict_dog_breed("temp.jpg")
    st.subheader(f"Predicted Breed: {breed}")