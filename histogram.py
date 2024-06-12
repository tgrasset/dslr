from sys import argv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    try:
        data = pd.read_csv("datasets/dataset_train.csv")
        subjects = ['Arithmancy', 'Astronomy', 'Herbology',
                    'Defense Against the Dark Arts', 'Divination', 'Muggle Studies',
                    'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions',
                    'Care of Magical Creatures', 'Charms', 'Flying']
        fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(30, 30))
        for i, ax in enumerate(axs.flatten()):
            if i < len(subjects):
                sns.histplot(data, x=subjects[i], hue='Hogwarts House', bins=20, multiple='dodge', common_norm=False, stat='percent', ax=ax)
            else:
                ax.axis('off')
        fig.savefig('histogram.png')

    except (AssertionError, Exception) as err:
        print("Error: ", err)

if __name__ == "__main__":
    main()
