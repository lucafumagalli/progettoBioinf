from sklearn.model_selection import StratifiedShuffleSplit
import os
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
from sanitize_ml_labels import sanitize_ml_labels
import compress_json
from typing import Tuple
from tensorflow.keras.utils import Sequence
from keras_mixed_sequence import MixedSequence
from keras_bed_sequence import BedSequence
from tensorflow.keras.callbacks import EarlyStopping


def report(promoters_labels_true:np.ndarray, promoters_labels_pred:np.ndarray)->np.ndarray:
    integer_metrics = accuracy_score, balanced_accuracy_score
    float_metrics = roc_auc_score, average_precision_score
    results1 = {
        sanitize_ml_labels(metric.__name__): metric(promoters_labels_true, np.round(promoters_labels_pred))
        for metric in integer_metrics
    }
    results2 = {
        sanitize_ml_labels(metric.__name__): metric(promoters_labels_true, promoters_labels_pred)
        for metric in float_metrics
    }
    return {
        **results1,
        **results2
    }

def precomputed(results, model:str, holdout:int)->bool:
    df = pd.DataFrame(results)
    if df.empty:
        return False
    return (
        (df.model == model) &
        (df.holdout == holdout)
    ).any()

def train(epigenomes, labels, models, kwargs, region, cell_line):
    epigenomes = epigenomes[region].values
    labels = labels[region]

    splits = 3
    holdouts = StratifiedShuffleSplit(n_splits=splits, test_size=0.2, random_state=42)

    if os.path.exists(cell_line + "/results_" + region + ".json"):
        results = compress_json.local_load(cell_line + "/results_" + region + ".json")
    else:
        results = []
        
    for i, (train, test) in tqdm(enumerate(holdouts.split(epigenomes, labels)), total=splits, desc="Computing holdouts", dynamic_ncols=True):
        for model, params in tqdm(zip(models, kwargs), total=len(models), desc="Training models", leave=False, dynamic_ncols=True):
            model_name = (
                model.__class__.__name__
                if model.__class__.__name__ != "Sequential"
                else model.name
            )
            if precomputed(results, model_name, i):
                continue
            model.fit(epigenomes[train], labels[train], **params)
            results.append({
                "model":model_name,
                "run_type":"train",
                "holdout":i,
                **report(labels[train], model.predict(epigenomes[train]))
            })
            results.append({
                "model":model_name,
                "run_type":"test",
                "holdout":i,
                **report(labels[test], model.predict(epigenomes[test]))
            })
            compress_json.local_dump(results, cell_line + "/results_" + region + ".json")

    df = pd.DataFrame(results)
    df = df.drop(columns=["holdout"])
    return df



def get_holdout(train:np.ndarray, test:np.ndarray, bed:pd.DataFrame, labels:np.ndarray, genome, batch_size=1024)->Tuple[Sequence, Sequence]:
    return (
        MixedSequence(
            x=BedSequence(genome, bed.iloc[train], batch_size=batch_size),
            y=labels[train],
            batch_size=batch_size
        ),
        MixedSequence(
            x= BedSequence(genome, bed.iloc[test], batch_size=batch_size),
            y=labels[test],
            batch_size=batch_size
        )
    )


def train_sequence(epigenomes, labels, genome, cell_line, region, models):

    bed = epigenomes[region].reset_index()[epigenomes[region].index.names]

    splits = 2
    holdouts = StratifiedShuffleSplit(n_splits=splits, test_size=0.2, random_state=42)
    
    if os.path.exists(cell_line + "/sequence_" + region + ".json"):
        results = compress_json.local_load(cell_line + "/sequence_" + region + ".json")
    else:
        results = []

    for i, (train_index, test_index) in tqdm(enumerate(holdouts.split(bed, labels[region])), total=splits, desc="Computing holdouts", dynamic_ncols=True):
        train, test = get_holdout(train_index, test_index, bed, labels[region], genome)
        for model in tqdm(models, total=len(models), desc="Training models", leave=False, dynamic_ncols=True):
            if precomputed(results, model.name, i):
                continue
            history = model.fit(
                train,
                steps_per_epoch=train.steps_per_epoch,
                validation_data=test,
                validation_steps=test.steps_per_epoch,
                epochs=100,
                shuffle=True,
                verbose=False,
                callbacks=[
                    EarlyStopping(monitor="val_loss", mode="min", patience=50),
                ]
            ).history
            scores = pd.DataFrame(history).iloc[-1].to_dict()
            results.append({
                "model":model.name,
                "run_type":"train",
                "holdout":i,
                **{
                    key:value
                    for key, value in scores.items()
                    if not key.startswith("val_")
                }
            })
            results.append({
                "model":model.name,
                "run_type":"test",
                "holdout":i,
                **{
                    key[4:]:value
                    for key, value in scores.items()
                    if key.startswith("val_")
                }
            })
            compress_json.local_dump(results, cell_line + "/sequence_" + region + ".json")
    
    df = pd.DataFrame(results)
    df = df.drop(columns=["holdout"])
    return df
