# Pokémon Type Classification Project

This is a beginner-level machine learning project that classifies Pokémon based on their stats using different classification algorithms.

## About the Project

The goal of this project is to predict the **primary type** (`Type 1`) of a Pokémon using basic stats such as HP, Attack, Defense, etc.  
This project was created as part of my learning journey in **Applied Artificial Intelligence** and aims to demonstrate my understanding of:
- Data cleaning and preprocessing
- Feature selection
- Model training and evaluation
- Using scikit-learn models

I’m still new to machine learning and Python development, but I’m actively learning and improving every day. 😊

## Dataset

The dataset used comes from a Pokémon CSV file that includes stats like:
- HP
- Attack
- Defense
- Sp. Atk
- Sp. Def
- Speed  
as well as each Pokémon’s `Type 1`.

Before training the models, the data is cleaned to:
- Remove rows with missing values
- Keep only types that appear at least 5 times (to ensure enough training data)

## Models Used

The following models were tested and compared:
- **K-Nearest Neighbors (KNN)** – `k=5`
- **Support Vector Machine (SVM)** – linear kernel
- **Random Forest Classifier** – 100 estimators

All models are evaluated using:
- **Accuracy score**
- **Classification report**
- **5-fold Cross-Validation**

## Results

Each model was trained and tested, and a classification report was generated.  
Here’s a general overview of what was observed:
- **Random Forest** had the best overall performance.
- **SVM** performed well with fewer assumptions.
- **KNN** worked decently but may be sensitive to scaling.

(You can run the code to see the exact accuracy values and classification reports.)

## How to Run

1. Clone the repository or download the files.
2. Make sure you have Python installed with the following libraries: pandas ; scikit-learn
3. Place the `pokemon_classifier.py` and the dataset CSV file in the same directory.
4. Run the script using your preferred IDE:
