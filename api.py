from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  
import joblib
import pandas as pd 

#load the modele
model = joblib.load('LightGBM_best_model.pkl')

#initiate the FastAPI
app = FastAPI()

#start the origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #replace * with the domain you want to allow "http://localhost:8000"
    allow_credentials=True, #allow credentials
    allow_methods=["*"],
    allow_headers=["*"],
)

#route to the root
@app.get('/')
def home():
    return {'message': 'API de prédiction des admissions IRA'}

#route de prediction
@app.post('/predict')
def predict(data: dict):
    try:
        #convert the data into dataframe
        data_df = pd.DataFrame(data)
        #make the prediction
        prediction = model.predict(data_df)
        #return the prediction
        return {'prediction': prediction[0]}
    except:
        return {'message': 'Erreur lors de la prédiction'}