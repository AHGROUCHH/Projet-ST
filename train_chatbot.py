import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import word_tokenize
import string
import joblib

# Télécharger les ressources de nltk
nltk.download('punkt')

# Charger les données
data = pd.read_csv('questions_reponses_rh.csv',encoding='ISO-8859-1')

# Prétraitement des données
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return ' '.join(tokens)

data['Question'] = data['Question'].apply(preprocess_text)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(data['Question'], data['Reponse'], test_size=0.2, random_state=42)

# Créer le pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Entraîner le modèle
pipeline.fit(X_train, y_train)

# Enregistrer le modèle
model_path = 'modele_chatbot.pkl'
joblib.dump(pipeline, model_path)

# Afficher un message pour confirmer l'enregistrement
print(f"Le modèle a été enregistré sous le nom '{model_path}'")

# Faire des prédictions
y_pred = pipeline.predict(X_test)

# Évaluer le modèle
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Fonction pour prédire la réponse à une nouvelle question
def get_response(question):
    question = preprocess_text(question)
    return pipeline.predict([question])[0]

# Interface de réponse aux questions
def respond_to_questions():
    print("Bonjour! Posez une question et j'essaierai d'y répondre.")
    while True:
        user_question = input("Votre question (ou tapez 'exit' pour quitter): ")
        if user_question.lower() == 'exit':
            print("Au revoir!")
            break
        response = get_response(user_question)
        print("Réponse:", response)

# Exemple d'utilisation
if __name__ == "__main__":
    respond_to_questions()
