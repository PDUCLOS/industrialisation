from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import lightgbm as lgb
import pandas as pd

# Charger le modèle (format texte LightGBM : pas de pickle, donc pas d'exécution de code au chargement).
# Exporté depuis LightGBM_best_model.pkl, toujours présent pour mémoire.
model = lgb.Booster(model_file="LightGBM_best_model.txt")
FEATURES = model.feature_name()

# Créer une instance FastAPI
app = FastAPI()


# Autoriser toutes les origines, méthodes et en-têtes (à adapter si besoin) pour eviter cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplacez "*" par ["http://localhost:8000"] pour plus de sécurité
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Définition de la route principale
@app.get("/")
def home():
    return {"message": "API de prédiction des admissions IRA", "variables": FEATURES}


# Route pour effectuer une prédiction
@app.post("/predict/")
def predict(data: dict):
    # LightGBM ne vérifie pas les noms de colonnes : un champ mal nommé serait lu « par position ».
    # On exige donc exactement les 28 variables du modèle, dans l'ordre du modèle.
    missing = sorted(set(FEATURES) - set(data))
    unknown = sorted(set(data) - set(FEATURES))
    if missing or unknown:
        raise HTTPException(422, {"variables_manquantes": missing, "variables_inconnues": unknown})
    try:
        df = pd.DataFrame([data])[FEATURES].astype("float64")
    except (TypeError, ValueError) as e:
        raise HTTPException(422, f"Valeur non numérique : {e}")

    prediction = model.predict(df)
    return {"prediction_ira": int(prediction[0])}
