from epigenomic_dataset import load_epigenomes
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
import pandas as pd
import os
import numpy as np
from ucsc_genomes_downloader import Genome


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

    # promoters_epigenomes = promoters_epigenomes.values
    # enhancers_epigenomes = enhancers_epigenomes.values

    # epigenomes = {
    #     "promoters": pd.DataFrame(promoters_epigenomes),
    #     "enhancers": pd.DataFrame(enhancers_epigenomes)
    # }

    # labels = {
    #     "promoters": pd.DataFrame(promoters_labels),
    #     "enhancers": pd.DataFrame(enhancers_labels)
    # }

    epigenomes = {
        "promoters": promoters_epigenomes,
        "enhancers": enhancers_epigenomes
    }

    labels = {
        "promoters": promoters_labels,
        "enhancers": enhancers_labels
    }

    return epigenomes, labels


def get_sequence():
    return Genome('hg19')

def get_sequence2(epigenomes, region):
    window_size = 200
    genome = Genome('hg19')
    sequences = {
        region: to_dataframe(
            flat_one_hot_encode(genome, data[:50000], window_size),
            window_size
        )
        for region, data in epigenomes.items()
    }


from keras_bed_sequence import BedSequence
def one_hot_encode(genome:Genome, data:pd.DataFrame, nucleotides:str="actg")->np.ndarray:
    return np.array(BedSequence(
        genome,
        bed=to_bed(data),
        nucleotides=nucleotides,
        batch_size=1
    ))

def to_bed(data:pd.DataFrame)->pd.DataFrame:
    """Return bed coordinates from given dataset."""
    return data.reset_index()[data.index.names]

def flat_one_hot_encode(genome:Genome, data:pd.DataFrame, window_size:int, nucleotides:str="actg")->np.ndarray:
    return one_hot_encode(genome, data, nucleotides).reshape(-1, window_size*4).astype(int)

def to_dataframe(x:np.ndarray, window_size:int, nucleotides:str="actg")->pd.DataFrame:
    return pd.DataFrame(
        x,
        columns = [
            f"{i}{nucleotide}"
            for i in range(window_size)
            for nucleotide in nucleotides
        ]
    )