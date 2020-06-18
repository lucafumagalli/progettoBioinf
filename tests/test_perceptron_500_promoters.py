from progettobioinf.models import perceptron, set_shape
from progettobioinf.retrieving_data import retrieving_data
from progettobioinf.train import train

def test_perceptron_500_enhancers():
    models = []
    kwargs = []
    epigenomes, labels = retrieving_data('GM12878')
    set_shape(epigenomes, 'promoters')
    model_perc, kwargs_perc = perceptron(500, 1024)
    models.append(model_perc)
    kwargs.append(kwargs_perc)
    train(epigenomes, labels, models, kwargs, 'promoters', 'GM12878')