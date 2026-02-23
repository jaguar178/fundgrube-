import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import pandas as pd

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="KI Bildklassifikation",
    page_icon="🧠",
    layout="centered"
)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("ℹ️ Informationen")
st.sidebar.write("""
Diese App verwendet ein trainiertes Keras-Modell zur Bildklassifikation.

**Bildanforderungen:**
- JPG, JPEG oder PNG
- Beliebige Größe (wird automatisch angepasst)

Das Bild wird auf 224x224 skaliert und normalisiert.
""")

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🧠 KI Bildklassifikation")
st.markdown("Lade ein Bild hoch und erhalte eine KI-Vorhersage.")

# -------------------------------------------------
# MODEL LOADING (CACHED)
# -------------------------------------------------
@st.cache_resource
def load_my_model():
    try:
        model = load_model("keras_Model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Fehler beim Laden des Modells: {e}")
        return None

model = load_my_model()

if model is None:
    st.stop()

# -------------------------------------------------
# LABELS LADEN
# -------------------------------------------------
try:
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except Exception as e:
    st.error(f"❌ Fehler beim Laden der Labels: {e}")
    st.stop()

# -------------------------------------------------
# FILE UPLOADER
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Bild auswählen...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Bild anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    # Bild vorbereiten
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)

    # -------------------------------------------------
    # VORHERSAGE
    # -------------------------------------------------
    with st.spinner("🔍 Modell analysiert das Bild..."):
        prediction = model.predict(data)

    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])

    # -------------------------------------------------
    # ERGEBNIS
    # -------------------------------------------------
    st.divider()
    st.subheader("🔎 Ergebnis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### 🏷️ Klasse")
        st.write(f"**{class_name}**")

    with col2:
        st.markdown("### 🎯 Sicherheit")
        if confidence_score > 0.8:
            st.success(f"{confidence_score * 100:.2f}%")
        elif confidence_score > 0.5:
            st.warning(f"{confidence_score * 100:.2f}%")
        else:
            st.error(f"{confidence_score * 100:.2f}%")

    # -------------------------------------------------
    # TOP 3
    # -------------------------------------------------
    st.subheader("🏆 Top 3 Vorhersagen")

    top_indices = prediction[0].argsort()[-3:][::-1]

    for i in top_indices:
        st.write(
            f"**{class_names[i]}** — {prediction[0][i] * 100:.2f}%"
        )

    # -------------------------------------------------
    # ALLE WAHRSCHEINLICHKEITEN
    # -------------------------------------------------
    st.subheader("📊 Wahrscheinlichkeitsverteilung")

    prob_df = pd.DataFrame({
        "Klasse": class_names,
        "Wahrscheinlichkeit (%)": prediction[0] * 100
    })

    st.bar_chart(prob_df.set_index("Klasse"))

    st.divider()
    st.caption("Entwickelt mit ❤️ und Streamlit")
