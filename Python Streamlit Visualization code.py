import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from urllib.parse import quote_plus
from sklearn.metrics import confusion_matrix

# --- MongoDB connection ---
username = "ccmt_user"
password = quote_plus("***********") # Replace with actual password
MONGO_URI = f"mongodb+srv://{username}:{password}@cluster0.o8r6odp.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client["ccmt"]
collection = db["metrics"]

st.title("🌱 CCMT Training Dashboard")
st.markdown("Visualizing crop disease & pest classification model performance")

# --- Fetch experiments ---
docs = list(collection.find().sort("timestamp", -1))
if not docs:
    st.warning("No experiments found in MongoDB yet!")
else:
    exp_names = [f"{d.get('experiment', 'unknown')} ({d['timestamp']})" for d in docs]
    choice = st.selectbox("Select Experiment:", exp_names)
    selected_doc = docs[exp_names.index(choice)]

    # --- Metrics summary ---
    st.subheader("📊 Metrics Overview")
    st.json({
        "Experiment": selected_doc.get("experiment"),
        "Epochs": selected_doc.get("epochs"),
        "Classes": selected_doc.get("num_classes"),
        "Best Val Accuracy": selected_doc.get("best_val_accuracy"),
        "Best Val Loss": selected_doc.get("best_val_loss"),
    })

    # --- Training Curves ---
    if "train_accuracy" in selected_doc:
        history_df = pd.DataFrame({
            "Epoch": list(range(1, selected_doc["epochs"] + 1)),
            "Train Accuracy": selected_doc.get("train_accuracy", []),
            "Val Accuracy": selected_doc.get("val_accuracy", []),
            "Train Loss": selected_doc.get("train_loss", []),
            "Val Loss": selected_doc.get("val_loss", [])
        })

        st.subheader("📈 Accuracy per Epoch")
        st.line_chart(history_df.set_index("Epoch")[["Train Accuracy", "Val Accuracy"]])

        st.subheader("📉 Loss per Epoch")
        st.line_chart(history_df.set_index("Epoch")[["Train Loss", "Val Loss"]])

    # --- Confusion Matrix ---
    if "y_true" in selected_doc and "y_pred" in selected_doc:
        st.subheader("🔎 Confusion Matrix")
        cm = confusion_matrix(selected_doc["y_true"], selected_doc["y_pred"])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
    else:
        st.info("No predictions stored for this experiment.")
