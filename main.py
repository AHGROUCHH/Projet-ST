from fastapi import FastAPI, Query
import joblib
from nltk.tokenize import word_tokenize
import nltk
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Charger les ressources de nltk
nltk.download('punkt')

# Charger le modèle
model_path = 'modele_chatbot.joblib'
pipeline = joblib.load(model_path)

# Prétraitement des données
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return ' '.join(tokens)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8081"],  # Permettre toutes les origines, pour les tests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle de données Pydantic pour la validation
class Question(BaseModel):
    question: str

# Endpoint pour obtenir des prédictions
@app.post("/predict")
async def predict(question: Question):
    question_processed = preprocess_text(question.question)
    response = pipeline.predict([question_processed])[0]
    return {"response": response}
