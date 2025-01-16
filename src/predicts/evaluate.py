#!/usr/bin/env python

import pickle
import pandas as pd
import numpy as np
import json
import click
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Ajouter le chemin du répertoire contenant data_split.py au sys.path
sys.path.append('./src/data')

from data_split import import_dataset

@click.command()
def evaluate_model():
    """ Evaluate the model using X_test_scaled and y_test, and save the scores and predictions to files. """
    # Demander les chemins des fichiers
    model_file = click.prompt('Entrer le chemin du fichier du modèle entraîné', type=click.Path(exists=True))
    model_file = f"{model_file}/gbr_model.pkl"

    input_filepath_X_test_scaled = click.prompt('Entrer le chemin du fichier X_test_scaled', type=click.Path(exists=True))
    input_filepath_X_test_scaled = f"{input_filepath_X_test_scaled}/X_test_scaled.csv"

    input_filepath_y_test = click.prompt('Entrer le chemin du fichier y_test', type=click.Path(exists=True))
    input_filepath_y_test = f"{input_filepath_y_test}/y_test.csv"

    scores_output_file = click.prompt('Entrer le chemin du fichier de sortie des scores', type=click.Path(exists=False))
    scores_output_file = f"{scores_output_file}/scores.json"

    predictions_output_file = click.prompt('Entrer le chemin du fichier de sortie des prédictions', type=click.Path(exists=False))
    predictions_output_file = f"{predictions_output_file}/predictions.csv"

    # Crée le dossier de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(scores_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(predictions_output_file), exist_ok=True)


    # Charger le modèle entraîné
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Import des datasets
    X_test_scaled = import_dataset(input_filepath_X_test_scaled)
    y_test = import_dataset(input_filepath_y_test)

    # Convertir y_test en une série ou un tableau 1D
    if isinstance(y_test, pd.DataFrame):
    	y_test = y_test.squeeze()

    # Faire des prédictions
    y_pred = model.predict(X_test_scaled)

    # Calculer les scores
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    scores = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

    # Enregistrer les scores dans un fichier JSON
    with open(scores_output_file, 'w') as f:
        json.dump(scores, f)

    # Enregistrer les prédictions dans un fichier CSV
    predictions = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    predictions.to_csv(predictions_output_file, index=False)

    print(f"Les scores ont été enregistrés dans {scores_output_file}")
    print(f"Les prédictions ont été enregistrées dans {predictions_output_file}")

if __name__ == '__main__':
    evaluate_model()
