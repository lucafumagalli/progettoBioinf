from models import *
from retrieving_data import retrieving_data
from data_analysis import *
if __name__ == "__main__":
    cell_line = 'A549'
    epigenomes, labels = retrieving_data(cell_line)
    class_rate_hist(epigenomes, labels, cell_line)
    drop_constant_features(epigenomes)
    robust_zscoring(epigenomes)
    run_correlation_tests(epigenomes, labels)
    extremely_correlated(epigenomes)