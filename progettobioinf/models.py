from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from keras_tqdm import TQDMNotebookCallback as ktqdm
from sklearn.tree import DecisionTreeClassifier
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import cpu_count
from tensorflow.keras.layers import Conv2D, Reshape
from tensorflow.keras.metrics import AUC


splits = 3
holdouts = StratifiedShuffleSplit(n_splits=splits, test_size=0.2, random_state=42)

shape_epigenomes = 0

def set_shape(epigenomes):
    global shape_epigenomes
    shape_epigenomes = epigenomes["promoters"].shape[1]

####### PERCEPTRON #######
def perceptron():
    perceptron = Sequential([
        Input(shape=(shape_epigenomes, )),
        Dense(1, activation="sigmoid")
    ], "Perceptron")

    perceptron.compile(
        optimizer="nadam",
        loss="binary_crossentropy"
    )

    kwargs = dict(
        epochs=600,
        batch_size=1024,
        validation_split=0.1,
        shuffle=True,
        verbose=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ktqdm(leave_outer=False)
        ]
    )
    return perceptron, kwargs

###### DECISION TREE #####
def decision_tree():
    decision_tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=50,
        random_state=42,
        class_weight="balanced"
    )
    kwargs = {}
    return decision_tree, kwargs

### MLP ###
def mlp():
    mlp = Sequential([
        Input(shape=(shape_epigenomes, )),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ], "MLP")

    mlp.compile(
        optimizer="nadam",
        loss="binary_crossentropy"
    )

    kwargs = dict(
        epochs=10,
        batch_size=1024,
        validation_split=0.1,
        shuffle=True,
        verbose=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ktqdm(leave_outer=False)
        ]
    )
    return mlp, kwargs

### RANDOM FOREST ###
def random_forest():
    random_forest = RandomForestClassifier(
        n_estimators=500,
        criterion="gini",
        max_depth=30,
        random_state=42,
        class_weight="balanced",
        n_jobs=cpu_count()
    )
    kwargs = {}
    return random_forest, kwargs
 
 ### FFNN ###
def ffnn():
    ffnn = Sequential([
        Input(shape=(shape_epigenomes, )),
        Dense(256, activation="relu"),
        Dense(128),
        BatchNormalization(),
        Activation("relu"),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ], "FFNN")

    ffnn.compile(
        optimizer="nadam",
        loss="binary_crossentropy"
    )

    kwargs = dict(
        epochs=10,
        batch_size=1024,
        validation_split=0.1,
        shuffle=True,
        verbose=False,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=50),
            ktqdm(leave_outer=False)
        ]
    )
    return ffnn, kwargs

## CNN ##
def cnn():
    cnn = Sequential([
        Input(shape=(200, 4)),
        Reshape((200, 4, 1)),
        Conv2D(64, kernel_size=(10, 2), activation="relu"),
        Conv2D(64, kernel_size=(10, 2), activation="relu"),
        Dropout(0.3),
        Conv2D(32, kernel_size=(10, 2), strides=(2, 1), activation="relu"),
        Conv2D(32, kernel_size=(10, 1), activation="relu"),
        Conv2D(32, kernel_size=(10, 1), activation="relu"),
        Dropout(0.3),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ], "CNN")

    cnn.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            AUC(curve="ROC", name="auroc"),
            AUC(curve="PR", name="auprc")
        ]
    )
    return cnn