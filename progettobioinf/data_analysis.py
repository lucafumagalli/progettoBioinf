import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import RobustScaler
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from minepy import MINE
import numpy as np
import seaborn as sns
from scipy.stats import entropy

p_value_threshold = 0.01
correlation_threshold = 0.05

def class_rate_hist(epigenomes, labels, cell_line):
    for region, x in epigenomes.items():
        print(
            f"The rate between features and samples for {region} data is: {x.shape[0]/x.shape[1]}"
        )
        print("="*80)

    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    for axis, (region, y) in zip(axes.ravel(), labels.items()):
        y.hist(ax=axis, bins=3)
        axis.set_title("Classes count in " + region)
    plt.savefig(cell_line + '/classes_rate.png')

def drop(df:pd.DataFrame)->pd.DataFrame:
    """Return DataFrame without constant features."""
    return df.loc[:, (df != df.iloc[0]).any()]

def drop_constant_features(epigenomes):
    for region, x in epigenomes.items():
        result = drop(x)
        if x.shape[1] != result.shape[1]:
            print(f"Features in {region} were constant and had to be dropped!")
            epigenomes[region] = result
        else:
            print(f"No constant features were found in {region}!")

def normalization(df:pd.DataFrame)->pd.DataFrame:
    return pd.DataFrame(
        RobustScaler().fit_transform(df.values),
        columns=df.columns,
        index=df.index
    )

def robust_zscoring(epigenomes):
    epigenomes = {
        region: normalization(x)
        for region, x in epigenomes.items()
    }

def pearson_test(epigenomes, labels, uncorrelated):
    for region, x in epigenomes.items():
        for column in tqdm(x.columns, desc=f"Running Pearson test for {region}", dynamic_ncols=True, leave=False):
            correlation, p_value = pearsonr(x[column].values.ravel(), labels[region].values.ravel())
            if p_value > p_value_threshold:
                print(region, column, correlation)
                uncorrelated[region].add(column)
    return uncorrelated
def spearman_test(epigenomes, labels, uncorrelated):
    for region, x in epigenomes.items():
        for column in tqdm(x.columns, desc=f"Running Spearman test for {region}", dynamic_ncols=True, leave=False):
            correlation, p_value = spearmanr(x[column].values.ravel(), labels[region].values.ravel())
            if p_value > p_value_threshold:
                print(region, column, correlation)
                uncorrelated[region].add(column)
    return uncorrelated
def mine_test(epigenomes, labels, uncorrelated):
    for region, x in epigenomes.items():
        for column in tqdm(uncorrelated[region], desc=f"Running MINE test for {region}", dynamic_ncols=True, leave=False):
            mine = MINE()
            mine.compute_score(x[column].values.ravel(), labels[region].values.ravel())
            score = mine.mic()
            if score < correlation_threshold:
                print(region, column, score)
            else:
                uncorrelated[region].remove(column)

def run_correlation_tests(epigenomes, labels):

    uncorrelated = {
        region: set()
        for region in epigenomes
    }
    pearson_test(epigenomes, labels, uncorrelated)
    spearman_test(epigenomes, labels, uncorrelated)
    mine_test(epigenomes, labels, uncorrelated)
    for region, x in epigenomes.items():
        epigenomes[region] =x.drop(columns=[
            col
            for col in uncorrelated[region]
            if col in x.columns
        ])

def extremely_correlated(epigenomes):
    p_value_threshold = 0.01
    correlation_threshold = 0.95
    extremely_correlated = {
        region: set()
        for region in epigenomes
    }
    scores = {
        region: []
        for region in epigenomes
    }

    for region, x in epigenomes.items():
        for i, column in tqdm(
            enumerate(x.columns),
            total=len(x.columns), desc=f"Running Pearson test for {region}", dynamic_ncols=True, leave=False):
            for feature in x.columns[i+1:]:
                correlation, p_value = pearsonr(x[column].values.ravel(), x[feature].values.ravel())
                correlation = np.abs(correlation)
                scores[region].append((correlation, column, feature))
                if p_value < p_value_threshold and correlation > correlation_threshold:
                    print(region, column, feature, correlation)
                    if entropy(x[column]) > entropy(x[feature]):
                        extremely_correlated[region].add(feature)
                    else:
                        extremely_correlated[region].add(column)

    for region, x in epigenomes.items():
        epigenomes[region] =x.drop(columns=[
            col
            for col in extremely_correlated[region]
            if col in x.columns
        ])
    scores = {
        region:sorted(score, key=lambda x: np.abs(x[0]), reverse=True)
        for region, score in scores.items()
    }
    return scores

def seaborn_plot_most_correlated(epigenomes, labels, scores, cell_line):
    for region, x in epigenomes.items():
        _, firsts, seconds = list(zip(*scores[region][:3]))
        columns = list(set(firsts+seconds))
        print(f"Most correlated features from {region} epigenomes")
        sns.pairplot(pd.concat([
            x[columns],
            labels[region],
        ], axis=1), hue=labels[region].columns[0])
        plt.savefig(cell_line + '/seaborn_plot_' + region +'_most.png')

def seaborn_plot_least_correlated(epigenomes, labels, scores, cell_line):
    for region, x in epigenomes.items():
        _, firsts, seconds = list(zip(*scores[region][-3:]))
        columns = list(set(firsts+seconds))
        print(f"Least correlated features from {region} epigenomes")
        sns.pairplot(pd.concat([
            x[columns],
            labels[region],
        ], axis=1), hue=labels[region].columns[0])
        plt.savefig(cell_line + '/seaborn_plot_' + region +'_least.png')