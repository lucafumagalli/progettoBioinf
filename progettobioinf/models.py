from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedShuffleSplit
from keras_tqdm import TQDMNotebookCallback as ktqdm

splits = 3
holdouts = StratifiedShuffleSplit(n_splits=splits, test_size=0.2, random_state=42)

models = []
kwargs = []

####### PERCEPTRON #######
perceptron = Sequential([
    Input(shape=(47, )),
    Dense(1, activation="sigmoid")
], "Perceptron")

perceptron.compile(
    optimizer="nadam",
    loss="binary_crossentropy"
)

models.append(perceptron)
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

###### DECISION TREE #####
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=50,
    random_state=42,
    class_weight="balanced"
)

models.append(decision_tree)
kwargs.append({})