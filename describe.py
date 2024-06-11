import pandas as pd
import numpy as np
from sys import argv

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

def ft_quantile(data, q):
    if q < 0 or q > 1:
        raise ValueError("q must be between 0 and 1")
    quantiles = {}
    for column in data.columns:
        feature = data[column].dropna()
        sorted_feature = sorted(list(feature))
        quantile_exact_pos = (len(feature) - 1) * q
        quantile_index = int(quantile_exact_pos)
        lower_quantile = sorted_feature[quantile_index]
        upper_quantile = sorted_feature[quantile_index + 1]
        interpolation_coef = quantile_exact_pos - quantile_index
        quantile = lower_quantile + (upper_quantile - lower_quantile) * interpolation_coef
        quantiles[column] = quantile
    return quantiles

def describe(dataframe):
    assert dataframe is not None, "The data set is not a proper csv file" 
    numerical_cols = dataframe.select_dtypes(include=['number']).columns
    non_empty_numerical_cols = [col for col in numerical_cols if not dataframe[col].isna().all()]
    numerical_data = dataframe[non_empty_numerical_cols]
    counts = ft_count(numerical_data)
    means = ft_mean(numerical_data)
    mins = ft_min(numerical_data)
    maxes = ft_max(numerical_data)
    stds = ft_std(numerical_data, means)
    quantile25 = ft_quantile(numerical_data, 0.25)
    quantile50 = ft_quantile(numerical_data, 0.50)
    quantile75 = ft_quantile(numerical_data, 0.75)
    return pd.DataFrame([counts, means, stds, mins, quantile25, quantile50, quantile75, maxes], index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"])

def main():
    try:
        assert len(argv) == 2, "The script needs a data file as argument"
        data = pd.read_csv(argv[1])
        print(describe(data))

    except (AssertionError, Exception) as err:
        print("Error: ", err)

if __name__ == "__main__":
    main()