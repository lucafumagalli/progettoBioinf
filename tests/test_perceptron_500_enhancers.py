from progettobioinf.models import perceptron, set_shape
from progettobioinf.retrieving_data import retrieving_data
from progettobioinf.train import train

models = []
kwargs = []
epigenomes, labels = retrieving_data('GM12878')
set_shape(epigenomes, 'enhancers')
model_perc, kwargs_perc = perceptron(200, 1024)
models.append(model_perc)
kwargs.append(kwargs_perc)
train(epigenomes, labels, models, kwargs, 'enhancers', 'GM12878')