#!/usr/bin/env python

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import logging
import click
from data_split import import_dataset

@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)

def main(input_filepath, output_filepath):
    """ Runs scaler scripts to turn clean data from (../preprocessed) into
        normalized data ready to be trained (saved in../preprocessed_scaled).
    """
    logger = logging.getLogger(__name__)
    logger.info('making data scaled and ready to be trained')

    input_filepath_X_train = click.prompt('Entrer le chemin du dossier pour le fichier de training à scaler', type=click.Path(exists=True))
    input_filepath_X_train = f"{input_filepath_X_train}/X_train.csv"
    input_filepath_X_test = click.prompt('Entrer le chemin du dossier pour le fichier de test à scaler', type=click.Path(exists=True))
    input_filepath_X_test = f"{input_filepath_X_test}/X_test.csv"

    output_filepath = click.prompt('Entrer le chemin du dossier pour enregistrer data scalés')

    X_train, X_test, X_train_scaled, X_test_scaled = data_scaled(input_filepath_X_train, input_filepath_X_test, output_filepath)
    return X_train, X_test, X_train_scaled, X_test_scaled

def data_scaled(input_filepath_X_train, input_filepath_X_test, output_filepath):
    # Import dataset
    X_train = import_dataset(input_filepath_X_train)
    X_test = import_dataset(input_filepath_X_test)

    # Initialisation du scaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Conversion des résultats en DataFrames pour une utilisation ultérieure
    save_dataframes_scaled(X_train_scaled, X_test_scaled, output_filepath)
    print("Les fichiers X_train_scaled, X_test_scaled sont enregistrés")
    return X_train, X_test, X_train_scaled, X_test_scaled

def save_dataframes_scaled(X_train_scaled, X_test_scaled, output_folderpath):
    # Crée le dossier de sortie s'il n'existe pas 
    os.makedirs(output_folderpath, exist_ok=True)
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
        pd.DataFrame(file).to_csv(output_filepath, index=False)

if __name__ == '__main__':
    main()
