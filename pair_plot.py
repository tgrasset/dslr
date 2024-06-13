import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    try:
        data = pd.read_csv("datasets/dataset_train.csv")
        data['House Slytherin'] = (data['Hogwarts House'] == 'Slytherin')
        data['House Hufflepuff'] = (data['Hogwarts House'] == 'Hufflepuff')
        data['House Gryffindor'] = (data['Hogwarts House'] == 'Gryffindor')
        data['House Ravenclaw'] = (data['Hogwarts House'] == 'Ravenclaw')
        data = data.drop(columns=['Hogwarts House', 'Index', 'First Name', 'Last Name', 'Birthday'])
        fig = sns.pairplot(data)
        fig.savefig("pair_plot.png")   
        
    except (AssertionError, Exception) as err:
        print("Error: ", err)

if __name__ == "__main__":
    main()