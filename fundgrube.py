import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Bildklassifikation", layout="centered")

st.title("🧠 KI Bildklassifikation")
st.write("Lade ein Bild hoch und erhalte die Vorhersage des Modells.")

# Modell laden (nur einmal)
@st.cache_resource
def load_my_model():
    return load_model("keras_Model.h5", compile=False)

model = load_my_model()

# Labels laden
with open("labels.txt", "r") as f:
    class_names = f.readlines()

# Bild-Upload
uploaded_file = st.file_uploader("Bild auswählen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Bild vorbereiten
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Ergebnis anzeigen
    st.subheader("🔎 Ergebnis")
    st.write(f"**Klasse:** {class_name}")
    st.write(f"**Sicherheit:** {confidence_score * 100:.2f}%")

    # Wahrscheinlichkeiten anzeigen
    st.subheader("📊 Alle Wahrscheinlichkeiten")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i].strip()}: {prob * 100:.2f}%")
