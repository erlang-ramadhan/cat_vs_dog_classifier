import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

st.title("🐾Cat vs Dog Classifier🐾")

model_path = hf_hub_download(
    repo_id="erlangram/catdog_model", 
    filename="cats_vs_dogs.keras")

model = tf.keras.models.load_model(model_path, compile=False)
st.write("Input shape model:", model.input_shape)

def preprocess(image):
    target_size = model.input_shape[1:3]
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

uploaded_file = st.file_uploader("You can upload an image or drag an image from browser", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    preview_placeholder = st.empty()
    preview_placeholder.image(image, caption="Preview Uploaded Image", use_container_width=True)

    if st.button("Predict"):

        preview_placeholder.empty()

        input_image = preprocess(image)
        pred = model.predict(input_image)

        if pred.shape[1] == 1:
            cat_prob = 1 - pred[0][0]
            dog_prob = pred[0][0]
            label = "Dog" if pred[0][0] > 0.5 else "Cat"
        else:
            cat_prob = pred[0][0]
            dog_prob = pred[0][1]
            label = "Cat" if np.argmax(pred[0]) == 0 else "Dog"

        if label == 'Cat':
            emoji = "🐱"
        else:
            emoji = "🐶"

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # Kolom 2: Hasil prediksi + pie chart
        with col2:
            st.subheader(f"It's a {label} {emoji}")

        st.write("")  
        st.write("")

        st.subheader("Prediction Confidence")

        labels = ['Cat', 'Dog']
        sizes = [cat_prob, dog_prob]
        colors = ['#FF9999','#66B2FF']

        chart1, chart2 = st.columns(2)


        with chart1:
            fig, ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, autopct='%1.1f%%',
            startangle=90, colors=colors, pctdistance=0.85)

            centre_circle = plt.Circle((0,0),0.70,fc='white')
            fig.gca().add_artist(centre_circle)
            ax.axis('equal')  
            st.pyplot(fig)

        with chart2:
            fig2, ax2 = plt.subplots()
            ax2.barh(labels, sizes, color=colors)
            ax2.set_xlim(0,1)
            for i, v in enumerate(sizes):
                ax2.text(v + 0.02, i, f"{v*100:.1f}%", va='center')
            ax2.set_xlabel("Probability")
            st.pyplot(fig2)