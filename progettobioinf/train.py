from sklearn.model_selection import StratifiedShuffleSplit
import os
from tqdm.auto import tqdm
import pandas as pd
import numpy as np

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

def train(epigenomes, labels, models, kwargs, region):
    epigenomes = epigenomes[region].values
    labels = labels[region].values
    splits = 3
    holdouts = StratifiedShuffleSplit(n_splits=splits, test_size=0.2, random_state=42)

    if os.path.exists("results.json"):
        results = compress_json.local_load("results.json")
    else:
        results = []
        
    for i, (train, test) in tqdm(enumerate(holdouts.split(epigenomes, labels)), total=splits, desc="Computing holdouts", dynamic_ncols=True):
        for model, params in tqdm(zip(models, kwargs), total=len(models), desc="Training models", leave=False, dynamic_ncols=True):
            print(type(epigenomes[train]))
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
            compress_json.local_dump(results, "results.json")

    df = pd.DataFrame(results)
    df = df.drop(columns=["holdout"])
