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
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as UTSNE
from multiprocessing import cpu_count
from prince import MFA


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
        y = pd.DataFrame(y)
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
        x = pd.DataFrame(x)
        for column in tqdm(x.columns, desc=f"Running Pearson test for {region}", dynamic_ncols=True, leave=False):
            #correlation, p_value = pearsonr(x[column].values.ravel(), labels[region].values.ravel())
            correlation, p_value = pearsonr(x[column].ravel(), labels[region].ravel())
            if p_value > p_value_threshold:
                print(region, column, correlation)
                uncorrelated[region].add(column)
    return uncorrelated
def spearman_test(epigenomes, labels, uncorrelated):
    for region, x in epigenomes.items():
        for column in tqdm(x.columns, desc=f"Running Spearman test for {region}", dynamic_ncols=True, leave=False):
            correlation, p_value = spearmanr(x[column].ravel(), labels[region].ravel())
            if p_value > p_value_threshold:
                print(region, column, correlation)
                uncorrelated[region].add(column)
    return uncorrelated
def mine_test(epigenomes, labels, uncorrelated):
    for region, x in epigenomes.items():
        for column in tqdm(uncorrelated[region], desc=f"Running MINE test for {region}", dynamic_ncols=True, leave=False):
            mine = MINE()
            mine.compute_score(x[column].ravel(), labels[region].ravel())
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
        labels2 = pd.DataFrame(labels[region])
        columns = list(set(firsts+seconds))
        print(f"Most correlated features from {region} epigenomes")
        sns.pairplot(pd.concat([
            x[columns].reset_index(drop=True),
            labels2.reset_index(drop=True),
        ], axis=1), hue=labels2.columns[0])
        plt.savefig(cell_line + '/seaborn_plot_' + region +'_most.png')

def seaborn_plot_least_correlated(epigenomes, labels, scores, cell_line):
    for region, x in epigenomes.items():
        _, firsts, seconds = list(zip(*scores[region][-3:]))
        labels2 = pd.DataFrame(labels[region])
        columns = list(set(firsts+seconds))
        print(f"Least correlated features from {region} epigenomes")
        sns.pairplot(pd.concat([
            x[columns].reset_index(drop=True),
            labels2.reset_index(drop=True),
        ], axis=1), hue=labels2.columns[0])
        plt.savefig(cell_line + '/seaborn_plot_' + region +'_least.png')

def get_top_most_different(epigenomes, labels, cell_line):
    top_number = 5
    for region, x in epigenomes.items():
        dist = euclidean_distances(x.T)
        most_distance_columns_indices = np.argsort(-np.mean(dist, axis=1).flatten())[:top_number]
        columns = x.columns[most_distance_columns_indices]
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
        print(f"Top {top_number} different features from {region}.")
        for column, axis in zip(columns, axes.flatten()):
            head, tail = x[column].quantile([0.05, 0.95]).values.ravel()
            
            mask = ((x[column] < tail) & (x[column] > head)).values
            
            cleared_x = x[column][mask]
            cleared_y = labels[region].ravel()[mask]
            
            cleared_x[cleared_y==0].hist(ax=axis, bins=20)
            cleared_x[cleared_y==1].hist(ax=axis, bins=20)

            axis.set_title(column)
        fig.tight_layout()
        plt.savefig(cell_line + '/feature_distribution.png')

def get_top_most_different_tuples(epigenomes, labels, cell_line):
    top_number = 5
    for region, x in epigenomes.items():
        dist = euclidean_distances(x.T)
        dist = np.triu(dist)
        tuples = list(zip(*np.unravel_index(np.argsort(-dist.ravel()), dist.shape)))[:top_number]
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
        print(f"Top {top_number} different tuples of features from {region}.")
        for (i, j), axis in zip(tuples, axes.flatten()):
            column_i = x.columns[i]
            column_j = x.columns[j]
            for column in (column_i, column_j):
                head, tail = x[column].quantile([0.05, 0.95]).values.ravel()
                mask = ((x[column] < tail) & (x[column] > head)).values
                x[column][mask].hist(ax=axis, bins=20, alpha=0.5)
            axis.set_title(f"{column_i} and {column_j}")
        fig.tight_layout()
        plt.savefig(cell_line + '/top_most_different.png')


def get_tasks(epigenomes, labels):
    tasks = {
        "x":[
            *[
                val
                for val in epigenomes.values()
            ],
        ],
        "y":[
            *[
                val.ravel()
                for val in labels.values()
            ],
        ],
        "titles":[
            "Epigenomes promoters",
            "Epigenomes enhancers",
        ]
    }
    xs = tasks["x"]
    ys = tasks["y"]
    titles = tasks["titles"]

    assert len(xs) == len(ys) == len(titles)

    for x, y in zip(xs, ys):
        assert x.shape[0] == y.shape[0]
    return xs, ys, titles

def pca(x:np.ndarray, n_components:int=2)->np.ndarray:
    return PCA(n_components=n_components, random_state=42).fit_transform(x)

def pca_plot(epigenomes, labels, cell_line):
    colors = np.array([
        "tab:blue",
        "tab:orange",
    ])

    xs, ys, titles = get_tasks(epigenomes, labels)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(32, 16))
    for x, y, title, axis in tqdm(zip(xs, ys, titles, axes.flatten()), desc="Computing PCAs", total=len(xs)):
        axis.scatter(*pca(x).T, s=1, color=colors[y])
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)
        axis.set_title(f"PCA decomposition - {title}", fontsize = 25)
    plt.savefig(cell_line + '/PCA.png')

def ulyanov_tsne(x:np.ndarray, perplexity:int, dimensionality_threshold:int=50, n_components:int=2):
    if x.shape[1] > dimensionality_threshold:
        x = pca(x, n_components=dimensionality_threshold)
    return UTSNE(n_components=n_components, perplexity=perplexity, n_jobs=cpu_count(), random_state=42, verbose=True).fit_transform(x)

def tsne_plot(epigenomes, labels, cell_line):
    colors = np.array([
        "tab:blue",
        "tab:orange",
    ])

    xs, ys, titles = get_tasks(epigenomes, labels)
    for perplexity in tqdm((30, 40, 50, 100, 500, 5000), desc="Running perplexities"):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(40, 20))
        for x, y, title, axis in tqdm(zip(xs, ys, titles, axes.flatten()), desc="Computing TSNEs", total=len(xs)):
            axis.scatter(*ulyanov_tsne(x, perplexity=perplexity).T, s=1, color=colors[y])
            axis.xaxis.set_visible(False)
            axis.yaxis.set_visible(False)
            axis.set_title(f"TSNE decomposition - {title}", fontsize = 25)
        fig.tight_layout()
        plt.savefig(cell_line + '/TSNE' + str(perplexity) + '.png')