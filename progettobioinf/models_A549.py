from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from keras_tqdm import TQDMNotebookCallback as ktqdm
from sklearn.tree import DecisionTreeClassifier

splits = 3
holdouts = StratifiedShuffleSplit(n_splits=splits, test_size=0.2, random_state=42)

models = []
kwargs = []

####### PERCEPTRON #######
def perceptron():
    perceptron = Sequential([
        Input(shape=(47, )),
        Dense(1, activation="sigmoid")
    ], "Perceptron")

    perceptron.compile(
        optimizer="nadam",
        loss="binary_crossentropy"
    )

    kwargs.append(dict(
        epochs=600,
        batch_size=1024,
        validation_split=0.1,
        shuffle=True,
        verbose=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ktqdm(leave_outer=False)
        ]
    ))
    return perceptron, kwargs

###### DECISION TREE #####
def decision_tree():
    decision_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=50,
        random_state=42,
        class_weight="balanced"
    )
    kwargs.append({})
    return decision_tree, kwargs

