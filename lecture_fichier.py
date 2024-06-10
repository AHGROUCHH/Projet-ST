import csv
import re

def preprocess_text(text):
    text = re.sub(r'[!@#$(),"%^*?:;~`0-9]', '', text)
    text = re.sub(r'\[|\]', ' ', text)  # Properly escape square brackets
    text = text.lower()
    return text

def process_csv(file_name):
    questions_answers = []
    with open(file_name, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if len(row) >= 2:  # Vérifier que la ligne a au moins deux éléments
                question = preprocess_text(row[0].strip())  # Supprimer les espaces blancs autour de la question
                answer = preprocess_text(row[1].strip())    # Supprimer les espaces blancs autour de la réponse
                questions_answers.append((question, answer))
    return questions_answers

def main():
    file_name = 'questions_reponses_rh.csv'
    questions_answers = process_csv(file_name)
    for idx, (question, answer) in enumerate(questions_answers, start=1):
        print(f"Question {idx}: {question}")
        print(f"Réponse {idx}: {answer}\n")

if __name__ == "__main__":
    main()
