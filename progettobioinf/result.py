from barplots import barplots
import pandas as pd
import os
from glob import glob
from PIL import Image
import PIL
import numpy as np
import shutil
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