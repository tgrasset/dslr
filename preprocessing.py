import numpy as np

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