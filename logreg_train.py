import pandas as pd
import sys
from preprocessing import SimpleImputer, StandardScaler, OneHotEncoder, PreprocessorPipeline
from sorting_hat import SortingHat, TARGET_CLASSES

COLS_TO_DROP = ['Index', 'First Name', 'Last Name', 'Birthday', 'Defense Against the Dark Arts', 'Arithmancy', 'Care of Magical Creatures']
NUMERICAL_COLS = ['Astronomy', 'Herbology', 'Divination', 'Muggle Studies', 'Ancient Runes', 'Charms', 'Potions', 'Transfiguration', 'History of Magic', 'Flying']
CATEGORICAL_COLS = ['Best Hand']
LEARNING_RATE = 0.02
EPOCHS = 1000


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
