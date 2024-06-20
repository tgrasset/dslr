import numpy as np
import pandas as pd
import json
import sys

COLS_TO_DROP = ['Index', 'First Name', 'Last Name', 'Birthday', 'Defense Against the Dark Arts', 'Arithmancy', 'Care of Magical Creatures']
NUMERICAL_COLS = ['Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes', 'Charms', 'Potions', 'Transfiguration', 'History of Magic', 'Flying']
CATEGORICAL_COLS = ['Best Hand']
TARGET_CLASSES = ['Slytherin', 'Hufflepuff', 'Gryffindor', 'Ravenclaw']
LEARNING_RATE = 0.02
EPOCHS = 1000

class SimpleImputer:
    def __init__(self, numerical_columns, categorical_columns):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.means = None
        self.modes = None

    def __repr__(self):
        return f"--- SimpleImputer ---\nMeans: {self.means}\nModes: {self.modes}\n"

    def fit(self, X):
        self.means = {column: X[column].mean() for column in self.numerical_columns}
        self.modes = {column: X[column].mode()[0] for column in self.categorical_columns}

    def transform(self, X):
        copy = X.copy()
        for column in self.numerical_columns:
            copy[column] = copy[column].fillna(self.means[column])
        for column in self.categorical_columns:
            copy[column] = copy[column].fillna(self.modes[column])
        return copy


class StandardScaler:
    def __init__(self, numerical_columns):
        self.numerical_columns = numerical_columns
        self.means = None
        self.stds = None

    def __repr__(self):
        return f"--- StandardScaler ---\nMeans: {self.means}\nStandard Deviations: {self.stds}\n"

    def fit(self, X):
        self.means = {column: X[column].mean() for column in self.numerical_columns}
        self.stds = {column: X[column].std() for column in self.numerical_columns}

    def transform(self, X):
        copy = X.copy()
        for column in self.numerical_columns:
            copy[column] = (copy[column] - self.means[column]) / self.stds[column]
        return copy


class OneHotEncoder:
    def __init__(self, categorical_columns, drop_last=True):
        self.categorical_columns = categorical_columns
        self.drop_last = drop_last
        self.new_columns = {}

    def __repr__(self):
        return f"--- OneHotEncoder ---\nColumns mapping: {self.new_columns}\nDrop Last: {self.drop_last}\n"

    def fit(self, X):
        for column in self.categorical_columns:
            self.new_columns[column] = []
            values = np.unique(X[column]).astype(str)
            if self.drop_last:
                values = values[:-1]
            for value in values:
                self.new_columns[column].append(column+'_'+value)

    def transform(self, X):
        copy = X.copy()
        for column in self.categorical_columns:
            for new_column in self.new_columns[column]:
                value = new_column.split('_')[-1]
                copy[new_column] = (copy[column].astype(str) == str(value)).astype(int)
        copy = copy.drop(columns=self.categorical_columns)
        return copy


class PreprocessorPipeline:
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def __repr__(self):
        repr = ""
        for preprocessor in self.preprocessors:
            repr = repr + preprocessor.__repr__() + '\n'
        return repr

    def fit(self, X):
        for preprocessor in self.preprocessors:
            preprocessor.fit(X)

    def transform(self, X):
        X_preprocessed = X.copy()
        for preprocessor in self.preprocessors:
            X_preprocessed = preprocessor.transform(X_preprocessed)
        return X_preprocessed


class LogReg:
    def __init__(self, n_features, target_column, lr):
        self.target_column = target_column
        self.weights = [0 for i in range(n_features)]
        self.bias = 0
        self.learning_rate = lr
        self.losses = []

    def __call__(self, X):
        linreg_result = self.bias + X @ self.weights
        return 1 / (1 + np.exp(-linreg_result))

    def predict(self, X):
        proba = self.__call__(X)
        return (proba >= 0.5).astype(int)

    def train_step(self, X_train, Y_train, X_test=None, Y_test=None, lr=None):
        if lr == None:
            lr = self.learning_rate
        error = self.__call__(X_train) - Y_train[self.target_column]
        self.weights = (self.weights - lr * (1 / len(X_train)) * (X_train.T @ error)).values.tolist()
        self.bias = self.bias - lr * (np.sum(error))
        loss = {
            'step': len(self.losses) + 1,
            'train_loss': self.loss(X_train, Y_train[self.target_column])
        }
        if X_test is not None and Y_test is not None:
            loss['test_loss'] = self.loss(X_test, Y_test[self.target_column])
        self.losses.append(loss)

    def loss(self, X, Y):
        Y_pred = self.__call__(X)
        loss =  - (1 / len(X)) * np.sum(Y * np.log(Y_pred) + (1 - Y) * np.log(1 - Y_pred))
        return loss


class SortingHat:
    def __init__(self, n_features, lr):

        self.logregs = {target_class: LogReg(n_features, target_class, lr) for target_class in TARGET_CLASSES}
        self.learning_rate = lr
        self.parameters = {
            logreg_item[0]: [logreg_item[1].bias] + logreg_item[1].weights for logreg_item in self.logregs.items()
        }
        self.losses = []

    def __call__(self, X):
        return {target_class: self.logregs[target_class].__call__(X) for target_class in TARGET_CLASSES}

    def predict(self, X):
        return pd.DataFrame(self.__call__(X)).idxmax(axis=1)

    def train_step(self, X_train, Y_train, X_test=None, Y_test=None, lr=None):
        if lr == None:
            lr = self.learning_rate
        for logreg in self.logregs.values():
            logreg.train_step(X_train, Y_train, X_test, Y_test, lr)
        loss = {
            'step': len(self.losses) + 1,
            'train_loss': sum([logreg.losses[-1]['train_loss'] for logreg in self.logregs.values()])
        }
        if X_test is not None and Y_test is not None:
            loss['test_loss'] = sum([logreg.losses[-1]['test_loss'] for logreg in self.logregs.values()])
        self.losses.append(loss)
        self.parameters = {
            logreg_item[0]: [logreg_item[1].bias] + logreg_item[1].weights for logreg_item in self.logregs.items()
        }

    def save_model(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.parameters, f, ensure_ascii=False, indent=4)

    def load_model(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.parameters = json.load(f)
        for logreg in self.logregs.values():
            logreg_parameters = self.parameters[logreg.target_column]
            logreg.weights = logreg_parameters[1:]
            logreg.bias = logreg_parameters[0]


def main():
    try:
        # Read and format data
        assert len(sys.argv) == 2, "The script needs a data file as argument"
        data = pd.read_csv(sys.argv[1])
        data = data.drop(columns=COLS_TO_DROP)
        X = data.drop(columns=['Hogwarts House']) # features
        for target_class in TARGET_CLASSES:
            data[target_class] = (data['Hogwarts House'] == target_class).astype(int)
        Y = data[TARGET_CLASSES] # target

        # Features preprocessings
        imputer = SimpleImputer(NUMERICAL_COLS, CATEGORICAL_COLS)
        scaler = StandardScaler(NUMERICAL_COLS)
        ohe = OneHotEncoder(CATEGORICAL_COLS)
        preprocessor = PreprocessorPipeline([imputer, scaler, ohe])
        preprocessor.fit(X)
        X_preprocessed = preprocessor.transform(X)

        # model training
        sorting_hat = SortingHat(X_preprocessed.shape[1], lr=LEARNING_RATE)
        for i in range(EPOCHS):
            sorting_hat.train_step(X_preprocessed, Y)
        sorting_hat.save_model('model.json')


    except (AssertionError, Exception) as err:
        print("Error: ", err)

if __name__ == "__main__":
    main()
