#!/usr/bin/env python

import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import os
import logging
import click
from data_split import import_dataset
import pickle
import numpy as np

@click.command()
def run_grid_search():
    """ Runs GridSearch scripts to search best Parameters to train with and
        registered best parameters into best_params.pkl file
    """
    logger = logging.getLogger(__name__)
    logger.info('GridSearch to find best parameters')

    input_filepath_X_train_scaled = click.prompt('Entrer le chemin du fichier X_train_scaled', type=click.Path(exists=True))
    input_filepath_X_train_scaled = f"{input_filepath_X_train_scaled}/X_train_scaled.csv"
    input_filepath_y_train = click.prompt('Entrer le chemin du fichier y_train', type=click.Path(exists=True))
    input_filepath_y_train = f"{input_filepath_y_train}/y_train.csv"

    params_file = click.prompt('Entrer le chemin du fichier paramètres yaml', type=click.Path(exists=True))
    params_file = f"{params_file}/params.yaml"

    best_params_file = click.prompt('Entrer le chemin du fichier des meilleurs paramètres pkl', type=click.Path(exists=False))
    best_params_file = f"{best_params_file}/best_params.pkl"

    # Crée le dossier de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(best_params_file), exist_ok=True)


    # Charger les paramètres depuis le fichier YAML
    with open(params_file, 'r') as file:
        params = yaml.safe_load(file)

    # Import des datasets
    X_train_scaled = import_dataset(input_filepath_X_train_scaled)
    y_train = import_dataset(input_filepath_y_train)

    # Aplatir y_train en utilisant ravel()
    y_train = np.ravel(y_train)

    # Initialiser le modèle
    model = RandomForestRegressor()

    # Initialiser GridSearchCV avec les paramètres chargés
    grid_search = GridSearchCV(estimator=model, param_grid=params['param_grid'], cv=5, n_jobs=-1, verbose=2)

    # Exécuter GridSearchCV
    grid_search.fit(X_train_scaled, y_train)

    # Enregistrer les meilleurs paramètres dans un fichier best_params.pkl
    with open(best_params_file, 'wb') as f:
        pickle.dump(grid_search.best_params_, f)

    print(f"Les meilleurs paramètres ont été enregistrés dans {best_params_file}")

if __name__ == '__main__':
    run_grid_search()
