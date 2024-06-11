import pandas as pd
import numpy as np

def ft_count(data):
    counts = {}
    for column in data.columns:
        count = 0
        for value in data[column]:
            if pd.notna(value):
                count += 1
        counts[column] = count
    return counts

def ft_mean(data):
    means = {}
    for column in data.columns:
        total = 0
        i = 0
        for value in data[column]:
            if pd.notna(value):
                total += value
                i += 1
        if i > 0:
            mean = total / i
        else:
            mean = float('nan')
        means[column] = mean
    return means

def ft_min(data):
    mins = {}
    for column in data.columns:
        min = float('inf')
        for value in data[column]:
            if value < min:
                min = value
        mins[column] = min
    return mins

def ft_max(data):
    maxes = {}
    for column in data.columns:
        max = float('-inf')
        for value in data[column]:
            if value > max:
                max = value
        maxes[column] = max
    return maxes

def ft_std(data, means):
    stds = {}
    for column in data.columns:
        values = data[column].dropna()
        if len(values) <= 1:
            stds[column] = float('nan')
        else:
            mean = means[column]
            variance = ((values - mean) ** 2).sum() / (len(values) - 1)
            stds[column] = np.sqrt(variance)
    return stds

def main():
    try:
        data = pd.read_csv("datasets/dataset_train.csv")
        assert data is not None, "The data set is not a proper csv file" 
        numerical_data = data.select_dtypes(include=['number']) 

        counts = ft_count(numerical_data)
        means = ft_mean(numerical_data)
        mins = ft_min(numerical_data)
        maxes = ft_max(numerical_data)
        stds = ft_std(numerical_data, means)

        res = pd.DataFrame([counts, means, mins, maxes, stds], index=["count", "mean", "min", "max", "std"])
        print (res)

    except (AssertionError, Exception) as err:
        print("Error: ", err)

if __name__ == "__main__":
    main()