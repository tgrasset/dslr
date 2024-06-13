import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    try:
        data = pd.read_csv("datasets/dataset_train.csv")
        data = data.drop(columns=['Index', 'First Name', 'Last Name'])
        features = data.drop(columns=['Hogwarts House'])
        features['Birthday'] = pd.to_datetime(features['Birthday'])
        features = features[['Best Hand'] + [col for col in features.columns if col != 'Best Hand']]
        fig, axs = plt.subplots(nrows=len(features.columns) - 1, ncols=len(features.columns) - 1, figsize=(200,200))

        for j, y in enumerate(features.columns):
            if j == 0:
                continue
            for i, x in enumerate(features.columns):
                if i == len(features.columns) - 1:
                    break
                if i >= j:
                    axs[i,j-1].axis('off')
                elif (x == 'Best Hand'):
                    sns.violinplot(data=features, x=x, y=y, ax=axs[i,j-1])
                else:
                    sns.scatterplot(data=features, x=x, y=y, ax=axs[i,j-1])
        fig.savefig('scatter_plot.png')
        
        
    except (AssertionError, Exception) as err:
        print("Error: ", err)

if __name__ == "__main__":
    main()