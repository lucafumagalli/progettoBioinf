from models import decision_tree, ffnn, mlp, random_forest, perceptron, cnn, set_shape
from retrieving_data import retrieving_data, get_sequence
from data_analysis import *
from train import train, train_sequence
from result import barplot

if __name__ == "__main__":
    cell_line = 'A549'
    models = []
    kwargs = []
    epigenomes, labels = retrieving_data(cell_line)
    # class_rate_hist(epigenomes, labels, cell_line)
    # drop_constant_features(epigenomes)
    # robust_zscoring(epigenomes)
    # run_correlation_tests(epigenomes, labels)
    # scores = extremely_correlated(epigenomes)
    # seaborn_plot_most_correlated(epigenomes, labels, scores, cell_line)
    # seaborn_plot_least_correlated(epigenomes, labels, scores, cell_line)
    # get_top_most_different(epigenomes, labels, cell_line)
    # get_top_most_different_tuples(epigenomes, labels, cell_line)
    #pca_plot(epigenomes, labels, cell_line)
    #tsne_plot(epigenomes, labels, cell_line)
    set_shape(epigenomes)
    modelperc, kwargsperc = perceptron()
    modeltree, kwargstree = decision_tree()
    modelmlp, kwargsmlp = mlp()
    modelffnn, kwargsffnn = ffnn()
    modelrf, kwargsrf = random_forest()
    #models.extend([modelperc, modeltree, modelmlp, modelffnn])
    models.extend([modelperc, modeltree])
    #kwargs.extend([kwargsperc, kwargstree, kwargsmlp, kwargsffnn])
    kwargs.extend([kwargsperc, kwargstree])
    df_promoters = train(epigenomes, labels, models, kwargs, 'promoters', cell_line)
    df_enhancers = train(epigenomes, labels, models, kwargs, 'enhancers', cell_line)
    barplot(df_promoters, cell_line, 'promoters')
    barplot(df_enhancers, cell_line, 'enhancers')

    #genome = get_sequence()
    #cnn = cnn()
    #models.append(cnn)
    #train_sequence(epigenomes, labels, genome, cell_line, 'promoters', models)