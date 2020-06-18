from progettobioinf.models import ffnn, set_shape
from progettobioinf.retrieving_data import retrieving_data
from progettobioinf.train import train

def test_FFNN_500_promoters():
    models = []
    kwargs = []
    epigenomes, labels = retrieving_data('GM12878')
    set_shape(epigenomes, 'promoters')
    model_ffnn, kwargs_ffnn = ffnn(500, 1024)
    models.append(model_ffnn)
    kwargs.append(kwargs_ffnn)
    train(epigenomes, labels, models, kwargs, 'promoters', 'GM12878')