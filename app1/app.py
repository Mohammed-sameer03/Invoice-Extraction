import streamlit as st
import easyocr
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import cv2

# ----------------------------
# 1️⃣ Load Model + Vectorizer + Label Encoder
# ----------------------------
@st.cache_resource
def load_model_and_tools():
    model = tf.keras.models.load_model(r"D:\pythonC\Brainwork Assignments\invoice\models\invoice_classifier.h5")
    vectorizer = joblib.load(r"D:\pythonC\Brainwork Assignments\invoice\models\tfidf_vectorizer.pkl")
    label_encoder = joblib.load(r"D:\pythonC\Brainwork Assignments\invoice\models\label_encoder.pkl")
    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_model_and_tools()

# ----------------------------
# 2️⃣ Initialize OCR
# ----------------------------
reader = easyocr.Reader(["en"], gpu=False)

# ----------------------------
# 3️⃣ Streamlit UI
# ----------------------------
st.set_page_config(page_title="Invoice Information Extractor", layout="wide")
st.title("🧾 Invoice Information Extraction using CNN (Text Classifier)")

uploaded_file = st.file_uploader("📤 Upload an invoice image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ✅ Convert to PIL Image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Invoice", use_container_width=True)

    # ----------------------------
    # 4️⃣ Extract text using EasyOCR (FIXED)
    # ----------------------------
    with st.spinner("🔍 Extracting text using OCR..."):
        # Convert PIL → NumPy (OpenCV format)
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Pass NumPy array directly to EasyOCR
        results = reader.readtext(image_np)

        # Extract detected text lines with confidence > 0.5
        extracted_texts = [res[1] for res in results if res[2] > 0.5]

    if len(extracted_texts) == 0:
        st.warning("⚠️ No text detected. Try uploading a clearer image.")
    else:
        st.subheader("📜 Extracted Text:")
        for t in extracted_texts:
            st.write("-", t)

        # ----------------------------
        # 5️⃣ Predict field type for each text line
        # ----------------------------
        X = vectorizer.transform(extracted_texts)
        preds = model.predict(X.toarray())
        pred_labels = label_encoder.inverse_transform(np.argmax(preds, axis=1))

        # ----------------------------
        # 6️⃣ Display predictions
        # ----------------------------
        st.subheader("🔍 Predicted Field Types:")
        for txt, label in zip(extracted_texts, pred_labels):
            st.markdown(f"**{label}** → {txt}")

        # ----------------------------
        # 7️⃣ (Optional) Summary
        # ----------------------------
        st.success("✅ Extraction and Classification completed successfully!")
