from models import decision_tree, ffnn, mlp, random_forest, perceptron, cnn, set_shape
from retrieving_data import retrieving_data, get_sequence, get_sequence2
from data_analysis import *
from train import train, train_sequence
from result import barplot, wilcoxon_test

if __name__ == "__main__":
    cell_line = 'A549'
    models = []
    kwargs = []
    epigenomes, labels = retrieving_data(cell_line)
    class_rate_hist(epigenomes, labels, cell_line)
    drop_constant_features(epigenomes)
    robust_zscoring(epigenomes)
    run_correlation_tests(epigenomes, labels)
    scores = extremely_correlated(epigenomes)
    seaborn_plot_most_correlated(epigenomes, labels, scores, cell_line)
    seaborn_plot_least_correlated(epigenomes, labels, scores, cell_line)
    get_top_most_different(epigenomes, labels, cell_line)
    get_top_most_different_tuples(epigenomes, cell_line)
    pca_plot(epigenomes, labels, cell_line)
    tsne_plot(epigenomes, labels, cell_line)

    for region in ['enhancers', 'promoters']:
        set_shape(epigenomes, region)
        modelperc, kwargsperc = perceptron(500, 1024)
        modeltree, kwargstree = decision_tree(500)
        modelmlp, kwargsmlp = mlp(500, 1024)
        modelffnn, kwargsffnn = ffnn(500, 1024)
        models.extend([modeltree, modelperc, modelmlp, modelffnn])
        kwargs.extend([kwargstree, kwargsperc, kwargsmlp, kwargsffnn])
        train_result = train(epigenomes, labels, models, kwargs, region, cell_line)
        barplot(train_result, cell_line, region)
        models.clear()
        kwargs.clear()
        print('Wilcoxon ' + region + ':')
        wilcoxon_test(train_result, 'FFNN', 'DecisionTreeClassifier')
        wilcoxon_test(train_result, 'FFNN', 'Perceptron')
        wilcoxon_test(train_result, 'FFNN', 'MLP')
        wilcoxon_test(train_result, 'Perceptron', 'DecisionTreeClassifier')
        wilcoxon_test(train_result, 'Perceptron', 'MLP')
        wilcoxon_test(train_result, 'MLP', 'DecisionTreeClassifier')

