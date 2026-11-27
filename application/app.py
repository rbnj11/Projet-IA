import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from detection import load_model, predict

st.set_page_config(page_title="Détection d'avions", layout="wide")
st.title("✈️ Détection et classification d'avions sur images satellites")


@st.cache_resource
def get_model():
    return load_model()


def draw(image, detections):
    img = np.array(image).copy()
    for d in detections:
        cv2.rectangle(img, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (255, 0, 0), 2)
        label = f'{d["classe"]} {d["confiance"]:.2f}'
        cv2.putText(img, label, (d["x1"], max(d["y1"] - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return img


model = get_model()
file = st.file_uploader("Choisis une image satellite", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    with st.spinner("Analyse en cours..."):
        detections = predict(model, image)

    col1, col2 = st.columns(2)
    col1.image(image, caption="Image originale", width="stretch")
    col2.image(draw(image, detections),
               caption=f"{len(detections)} avions détectés", width="stretch")

    if detections:
        df = pd.DataFrame(detections)
        st.subheader("Détections")
        st.dataframe(df, width="stretch")
        st.subheader("Répartition par classe")
        st.bar_chart(df["classe"].value_counts())
        st.download_button("Télécharger les résultats (CSV)",
                           df.to_csv(index=False), "detections.csv")
    else:
        st.warning("Aucun avion détecté.")
        
#cd "C:\Users\rbnja\Downloads\Projet ia\application"
#python -m streamlit run app.py