import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load  # Importer dump
import nltk
from nltk.tokenize import word_tokenize
import os
# Télécharger les ressources de nltk
nltk.download('punkt')

# Charger les données
data = pd.read_csv('questions_reponses_rh.csv', encoding='utf-8')

# Prétraitement des données
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    return ' '.join(tokens)

data['Question'] = data['Question'].apply(preprocess_text)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(data['Question'], data['Reponse'], test_size=0.2, random_state=42)

# Créer et entraîner le modèle
tfidf_vectorizer = TfidfVectorizer()
logistic_regression = LogisticRegression(max_iter=1000)
pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('clf', logistic_regression)
])
pipeline.fit(X_train, y_train)

# Faire des prédictions
y_pred = pipeline.predict(X_test)

# Évaluer le modèle
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Sauvegarde du modèle
dump(pipeline, 'modele_chatbot.joblib')
if os.path.exists('modele_chatbot.joblib'):

    print("Le modèle a été sauvegardé avec succès sous le nom 'modele_chatbot.joblib'.")
else:
    print("Erreur: Le fichier 'modele_chatbot.joblib' n'a pas été trouvé. La sauvegarde du modèle a échoué.")

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
