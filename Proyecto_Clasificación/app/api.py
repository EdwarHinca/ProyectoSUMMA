from fastapi import FastAPI
from pydantic import BaseModel
from app.utils import LabelEncoderTransformer
import dill
import pandas as pd
import numpy as np
from typing import List

# Inicializar FastAPI
app = FastAPI()

# Registrar explícitamente la clase en el espacio global
globals()['LabelEncoderTransformer'] = LabelEncoderTransformer

# Cargar el modelo
with open('model/model_xg.pkl', 'rb') as f:
    model = dill.load(f)

# Estructura del JSON de entrada
class ModelInput(BaseModel):
    autoID: str
    SeniorCity: int
    Partner: str
    Dependents: str
    Service1: str
    Service2: str
    Security: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    Charges: float
    

# Diccionario de mapeo para la salida
output_map = {0: "Alpha", 1: "Betha"}

# Ruta principal para la predicción
@app.post("/predict")
def predict(input_data: List[ModelInput]):
    try:
        results=[]
        for data in input_data:
                
            input_df = pd.DataFrame([[
                data.Security,
                data.OnlineBackup,
                data.TechSupport,
                data.Contract,
                data.PaymentMethod,
                data.Charges,
            ]], columns=['Security', 'OnlineBackup', 'TechSupport', 'Contract', 'PaymentMethod', 'Charges'])

            prediction = model.predict(input_df)
            # Mapear la salida
            predicted_class = output_map.get(prediction[0], "Unknown")
            results.append({"class": predicted_class})

        return {"predictions": results}

    except Exception as e:
        return {"error": str(e)}
