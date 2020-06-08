from epigenomic_dataset import load_epigenomes
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
import pandas as pd
import os

def retrieving_data(cell_line, window_size=200):
    try:
        os.mkdir(cell_line)
    except OSError:
        print ("Directory already exists")
    else:
        print ("Successfully created the directory for cell line")

    promoters_epigenomes, promoters_labels = load_epigenomes(
        cell_line = cell_line,
        dataset = "fantom",
        regions = "promoters",
        window_size = window_size
    )

    enhancers_epigenomes, enhancers_labels = load_epigenomes(
        cell_line = cell_line,
        dataset = "fantom",
        regions = "enhancers",
        window_size = window_size
    )

    promoters_epigenomes = promoters_epigenomes.droplevel(1, axis=1) 
    enhancers_epigenomes = enhancers_epigenomes.droplevel(1, axis=1) 

    promoters_labels = promoters_labels.values.ravel()
    enhancers_labels = enhancers_labels.values.ravel()

    # Imputation of NaN Values
    promoters_epigenomes[promoters_epigenomes.columns] = KNNImputer(n_neighbors=promoters_epigenomes.shape[0]//10).fit_transform(promoters_epigenomes)
    enhancers_epigenomes[enhancers_epigenomes.columns] = KNNImputer(n_neighbors=enhancers_epigenomes.shape[0]//10).fit_transform(enhancers_epigenomes)

    # Robust normalization of the values
    # promoters_epigenomes[promoters_epigenomes.columns] = RobustScaler().fit_transform(promoters_epigenomes)
    # enhancers_epigenomes[enhancers_epigenomes.columns] = RobustScaler().fit_transform(enhancers_epigenomes)

    promoters_epigenomes = promoters_epigenomes.values
    enhancers_epigenomes = enhancers_epigenomes.values

    epigenomes = {
        "promoters": pd.DataFrame(promoters_epigenomes),
        "enhancers": pd.DataFrame(enhancers_epigenomes)
    }

    labels = {
        "promoters": pd.DataFrame(promoters_labels),
        "enhancers": pd.DataFrame(enhancers_labels)
    }

    return epigenomes, labels
