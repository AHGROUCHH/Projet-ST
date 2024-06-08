import csv

def process_csv(file_name):
    questions_answers = []
    with open(file_name, 'r', encoding='latin-1') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if len(row) >= 2:  # Vérifier que la ligne a au moins deux éléments
                question = row[0].strip()  # Supprimer les espaces blancs autour de la question
                answer = row[1].strip()    # Supprimer les espaces blancs autour de la réponse
                questions_answers.append((question, answer))
    return questions_answers

def main():
    file_name = 'questions_reponses_rh.csv'
    questions_answers = process_csv(file_name)
    print(questions_answers)

if __name__ == "__main__":
    main()
