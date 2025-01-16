import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
import click

@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in../preprocessed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Split des données raw de départ')

    input_filepath = click.prompt('Entrer le chemin du dossier dans lequel se trouve le raw.csv', type=click.Path(exists=True))
    input_filepath = f"{input_filepath}/raw.csv"
    output_filepath = click.prompt("Entrer le chemin du dossier d'enregistrement des data splittées", type=click.Path())

    df, X, y, X_train, X_test, y_train, y_test = process_data(input_filepath, output_filepath)
    return df, X, y, X_train, X_test, y_train, y_test

def process_data(input_filepath, output_filepath):
    # Import datasets
    df = import_dataset(input_filepath)
    print("DataFrame raw.csv loaded")
    #print(df.head().to_string()) # Affiche les premières lignes du DataFrame pour vérifier le chargement

    # Supprime la colonne date qui n'apporte pas d'information
    df = delete_date(df)
    #print(df.head().to_string())

    # Création des grands jeux de données X et y
    X = drop_column_silica_concentrate(df)
    y = target_silica_concentrate(df)

    # Création des 4 dataframes de base
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Sauvegarde des dataframes
    save_dataframes(X_train, X_test, y_train, y_test, output_filepath)
    print("Les fichiers X_train, X_test, y_train, y_test sont enregistrés")
    return df, X, y, X_train, X_test, y_train, y_test

# Import des données
def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

# Suppression de la colonne date
def delete_date(df):
    if 'date' in df.columns:
        #print("Colonne date supprimée:")
        df = df.drop('date', axis=1)
    else:
        print("La colonne 'date' n'existe pas dans le DataFrame.")
    return df

# Récupération d'un dataframe sans la variable cible
def drop_column_silica_concentrate(df):
    X = df.drop('silica_concentrate', axis=1)
    return X

# Récupération d'un dataframe avec uniquement la variable cible
def target_silica_concentrate(df):
    y = df['silica_concentrate']
    return y

# Split en 4 dataframes à partir de X et y
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Sauvegarde des dataframes
def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
	os.makedirs(output_folderpath, exist_ok=True) 
	for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']): 
		output_filepath = os.path.join(output_folderpath, f'{filename}.csv') 
		print(f"Saving {filename} to {output_filepath}") 
		file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    main()
