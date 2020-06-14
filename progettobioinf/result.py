from barplots import barplots
import pandas as pd
import os
from glob import glob
from PIL import Image
import PIL
import numpy as np
import shutil
from scipy.stats import wilcoxon

def barplot(df:pd.DataFrame, cell_line, region):

    barplots(
        df,
        groupby=["model", "run_type"],
        show_legend=False,
        height=5,
        orientation="horizontal"
    )
    try:
        os.mkdir(cell_line)
    except OSError:
        print ("Directory already exists")
    else:
        print ("Successfully created the directory for cell line")

    list_im = []
    for filename in glob('barplots/*.png'):
        list_im.append(filename)
    imgs = [ PIL.Image.open(i) for i in list_im ]
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    img_np_array = [np.asarray(i.resize(min_shape)) for i in imgs]
    imgs_comb = np.vstack(img_np_array)
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb.save(cell_line + '/models_scores_' + region + '.png')
    shutil.rmtree('barplots', ignore_errors=True)

def wilcoxon_test(df:pd.DataFrame):

    # Here we will be doing a statistical test.
    models = df[
        (df.run_type == "test")
    ]
    perceptron_scores = models[models.model=="Perceptron"]
    dectree_scores = models[models.model=="DecisionTreeClassifier"]
    alpha = 0.01
    for metric in perceptron_scores.columns[-4:]:
        print(metric)
        a,  b = perceptron_scores[metric], dectree_scores[metric]
        stats, p_value = wilcoxon(a, b)
        if p_value > alpha:
            print(p_value, "The two models performance are statistically identical.")
        else:
            print(p_value, "The two models performance are different")
            if a.mean() > b.mean():
                print("The first model is better")
            else:
                print("The second model is better")