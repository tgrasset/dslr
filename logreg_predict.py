import pandas as pd
import sys
from preprocessing import SimpleImputer, StandardScaler, OneHotEncoder, PreprocessorPipeline
from sorting_hat import SortingHat

COLS_TO_DROP = ['Index', 'First Name', 'Last Name', 'Birthday', 'Defense Against the Dark Arts', 'Arithmancy', 'Care of Magical Creatures', 'Hogwarts House']
NUMERICAL_COLS = ['Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes', 'Charms', 'Potions', 'Transfiguration', 'History of Magic', 'Flying']
CATEGORICAL_COLS = ['Best Hand']
LEARNING_RATE = 0.02
EPOCHS = 1000


def main():
    try:
        # Read and keep features
        assert len(sys.argv) == 3, "The script needs a data test file and a model file as arguments"
        data_test = pd.read_csv(sys.argv[1])
        X_test = data_test.drop(columns=COLS_TO_DROP)

        # Features preprocessing
        data_train = pd.read_csv("datasets/dataset_train.csv")
        X_train = data_train.drop(columns=COLS_TO_DROP)
        imputer = SimpleImputer(NUMERICAL_COLS, CATEGORICAL_COLS)
        scaler = StandardScaler(NUMERICAL_COLS)
        ohe = OneHotEncoder(CATEGORICAL_COLS)
        preprocessor = PreprocessorPipeline([imputer, scaler, ohe])
        preprocessor.fit(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)

        # Prediction
        sorting_hat = SortingHat(X_test_preprocessed.shape[1], lr=LEARNING_RATE)
        sorting_hat.load_model(sys.argv[2])
        houses = {0 : 'Slytherin', 1 : 'Hufflepuff', 2 : 'Gryffindor', 3 : 'Ravenclaw'}
        X_test_preprocessed['Hogwarts House'] = sorting_hat.predict(X_test_preprocessed).replace(houses)
        res = pd.DataFrame({'Index': X_test_preprocessed.index, 'Hogwarts House': X_test_preprocessed['Hogwarts House']})
        res.to_csv('houses.csv', index=False)
        
    except (AssertionError, Exception) as err:
        print("Error: ", err)

if __name__ == "__main__":
    main()