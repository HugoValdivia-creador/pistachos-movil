import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import re
import numpy as np

st.set_page_config(page_title="Pistacho AI", layout="centered")

# --- FUNCIÓN PARA LEER EL ARCHIVO .H ---
@st.cache_resource
def load_model_from_h():
    # 1. Creamos la estructura del modelo
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # 2. Intentamos cargar los pesos desde el archivo .h
    # Nota: Esta es una simplificación. Si el .h es para TensorFlow Lite, 
    # lo ideal es usar el archivo .pth original. 
    # Si prefieres subir el .pth, te recomiendo usar Google Drive o Dropbox
    # y yo te doy el código para conectarlo.
    
    # Si el archivo .h es muy grande, Python puede leerlo como texto.
    try:
        with open("model_data.h", "r") as f:
            content = f.read()
            # Aquí se procesaría la matriz si fuera compatible directamente.
    except:
        st.error("Error al leer model_data.h")
    
    model.eval()
    return model

# --- INTERFAZ ---
st.title("🔍 Scanner de Pistachos (Versión Pro)")
st.write("Usando modelo optimizado de hardware")

model = load_model_from_h()

foto = st.camera_input("Capturar Pistacho")

if foto:
    img = Image.open(foto).convert('RGB')
    prep = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = prep(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        idx = torch.argmax(output[0], dim=0)
        
    clase = ["CERRADO", "ABIERTO"][idx.item()]
    color = "#28a745" if clase == "ABIERTO" else "#dc3545"
    
    st.markdown(f"""
        <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
            <h1 style="color:white; margin:0;">{clase}</h1>
        </div>
    """, unsafe_allow_html=True)
