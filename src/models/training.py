#!/usr/bin/env python

import yaml
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle
import numpy as np
import click
import sys
import os

# Ajouter le chemin du répertoire contenant data_split.py au sys.path
sys.path.append('./src/data')

from data_split import import_dataset

@click.command()

def train_model():
    """ Train a RandomForestRegressor model using the best parameters from best_params.pkl
        and save the trained model to a file.
    """
    # Demander les chemins des fichiers
    input_filepath_X_train_scaled = click.prompt('Entrer le chemin du fichier X_train_scaled', type=click.Path(exists=True))
    input_filepath_X_train_scaled = f"{input_filepath_X_train_scaled}/X_train_scaled.csv"
    input_filepath_y_train = click.prompt('Entrer le chemin du fichier y_train', type=click.Path(exists=True))
    input_filepath_y_train = f"{input_filepath_y_train}/y_train.csv"

    best_params_file = click.prompt('Entrer le chemin du fichier des meilleurs paramètres pkl', type=click.Path(exists=True))
    best_params_file = f"{best_params_file}/best_params.pkl"

    model_output_file = click.prompt('Entrer le chemin du fichier de sortie du modèle entraîné', type=click.Path(exists=False))
    model_output_file = f"{model_output_file}/gbr_model.pkl"

    # Crée le dossier de sortie s'il n'existe pas 
    os.makedirs(os.path.dirname(model_output_file), exist_ok=True)

    # Charger les meilleurs paramètres depuis le fichier best_params.pkl
    with open(best_params_file, 'rb') as f:
        best_params = pickle.load(f)

    # Import des datasets
    X_train_scaled = import_dataset(input_filepath_X_train_scaled)
    y_train = import_dataset(input_filepath_y_train)

    # Aplatir y_train en utilisant ravel()
    y_train = np.ravel(y_train)

    # Initialiser le modèle avec les meilleurs paramètres
    model = RandomForestRegressor(**best_params)

    # Entraîner le modèle
    model.fit(X_train_scaled, y_train)

    # Enregistrer le modèle entraîné dans un fichier
    with open(model_output_file, 'wb') as f:
        pickle.dump(model, f)

    print(f"Le modèle a été entraîné et enregistré dans {model_output_file}")

if __name__ == '__main__':
    train_model()
