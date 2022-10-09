# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.8.10 ('ml4t')
#     language: python
#     name: python3
# ---

# + [markdown] pycharm={"name": "#%% md\n"} id="sNw45BR0m62X"
# # Importing dependencies
# We used the following libraries in the first task:
# - `pandas` to manipulate the data
# - `scikit-learn` for imputing and scaling the data
# - `seaborn` and `matplotlib` for visualization.

# + pycharm={"name": "#%%\n"} id="4CRS48qom62Z"
import pandas as pd
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# + [markdown] pycharm={"name": "#%% md\n"} id="XyY6wwS-m62a"
# # Data exploration
# Let's load the training dataset from the corresponding .csv file.
# Since we know that the columns represent the mean/standard deviation of the positions and angles of the 60 points, respectively, followed by the label name and code, let us rename the columns accordingly to allow for easier reading.

# + pycharm={"name": "#%%\n"} id="KNoDyk_fm62a"
def load_data():
  training_data = pd.read_csv('train-final.csv', header=None)

  name_mappings = {
      # Feature columns
      **{i:f'positions_mean_{i}' for i in range(60)},
      **{i:f'positions_std_{i}' for i in range(60, 120)},
      **{i:f'angles_mean_{i}' for i in range(120, 180)},
      **{i:f'angles_std_{i}' for i in range(180, 240)},
      # Label columns
      **{240: 'label_name', 241: 'label_code'},
  }

  training_data.rename(name_mappings, axis=1, inplace=True)
  training_feature_columns = training_data.columns[:-2]

  training_features = training_data[training_feature_columns]
  training_labels = training_data.label_name

  return training_features, training_labels


training_features, training_labels = load_data()

# + [markdown] pycharm={"name": "#%% md\n"} id="_aYlx2ckm62b"
# Let's show some of the data

# + pycharm={"name": "#%%\n"} id="paURPIp3m62b" outputId="4b22de09-f76e-4f25-8e2d-9d356e7ca70d" colab={"base_uri": "https://localhost:8080/", "height": 300}
training_features.head()

# + [markdown] pycharm={"name": "#%% md\n"} id="DYPTNtFmm62c"
# How many different labels do we have in the training dataset?

# + pycharm={"name": "#%%\n"} id="l8hDw1bgm62c" outputId="7f333633-1d1c-4483-8fbc-0c1f9acf96a4" colab={"base_uri": "https://localhost:8080/"}
number_of_classes = training_labels.nunique()
number_of_classes

# + [markdown] pycharm={"name": "#%% md\n"} id="2NwxuXvqm62c"
# Now let's take a look at how many occurrences we have of each label.

# + pycharm={"name": "#%%\n"} id="_O-CkrFfm62d" outputId="ad9afcd5-4836-4ad1-fa9b-7d7a5aa5954e" colab={"base_uri": "https://localhost:8080/", "height": 322}
training_labels.value_counts().plot(kind='bar', figsize=(10, 4))

# + [markdown] pycharm={"name": "#%% md\n"} id="YlISrYJpm62d"
# We can see that `child` is the most common label in the training dataset and that `go` is the least common label.

# + [markdown] pycharm={"name": "#%% md\n"} id="AhOTmkwRm62d"
# Now let's look for columns that have missing values. The missing values are in the following columns (along with the missing value count):
#

# + pycharm={"name": "#%%\n"} id="rTfhsEE8m62f" colab={"base_uri": "https://localhost:8080/"} outputId="62eb3157-0b72-4f9c-a1bf-91cf110ced34"
# Look for columns that have missing values
columns_null_sum = training_features.isnull().sum()
columns_with_nulls = columns_null_sum[columns_null_sum > 0]

print(
    "Total amount of missing values in the dataframe:", 
    training_features.isnull().sum().sum()
)
print(
    "Missing values in the following column indexes (and missing value count):"
)
print(columns_with_nulls)


# + [markdown] pycharm={"name": "#%% md\n"} id="yprFNiVkm62j"
# Some classifiers are more sensitive to the range, mean & outliers of the features, such as linear regression models, for example.
# In order to be able to train a wide range of classifiers and compare them, we will need to preprocess the data for scaling and outlier treatment.

# + [markdown] pycharm={"name": "#%% md\n"} id="t9yB47dLm62k"
# Let's see if the dataset also contains outliers. There are quite a few way to detect outliers (Source:
# [Outlier detection methods in Scikit-Learn](https://scikit-learn.org/stable/modules/outlier_detection.html)):
# - Isolation forest
# - Local outlier factor
# - One-class support vector machine (SVM)
# - Elliptic envelope
#
# We start by doing a boxplot for all features to get a visual indication of the outlier situation.

# + pycharm={"name": "#%%\n"} id="r77J-ZiOm62k" outputId="0bddf0cf-5a7c-4708-823f-a73359d67ac3" colab={"base_uri": "https://localhost:8080/", "height": 337}
training_features.boxplot(figsize=(18,7))
plt.xticks([1], [''])

# + [markdown] id="m4tCX_XRnYRS" pycharm={"name": "#%% md\n"}
# Based on the boxplot, there appears to be many columns with outliers. Many classifiers, e.g. linear classifiers like Logistic Regression will not handle outliers well, so we need to find a way to handle also outliers.

# + [markdown] id="J1sVekRXoT4Q" pycharm={"name": "#%% md\n"}
# # Methods

# + [markdown] id="Kq-ne6mOolOy" pycharm={"name": "#%% md\n"}
# ## Outliers
#
# As we saw in the boxplot above, there are many columns with outliers. And while there are many methods to detect outliers,let's begin with just identifying the values that are farthest from the mean. 
#
# A simple approach is to identify the values that lie outside of 3$\sigma$ (that is, three times the standard deviation) as outliers, and drop the rows that have at least one outlier. Let's give it a try.

# + pycharm={"name": "#%%\n"} id="sOaxwVV_m62k"
#training_features_outliers_marked = training_features[abs(training_features) <= 3]
from scipy import stats

training_features_outliers_marked = training_features[
    np.abs(stats.zscore(training_features.fillna(training_features.mean()))) < 3
]

# + colab={"base_uri": "https://localhost:8080/", "height": 300} id="gGFZoxbwpqaE" outputId="e4bbb462-72f6-46b3-e83a-1d5e98fe2561" pycharm={"name": "#%%\n"}
training_features_outliers_marked.head()

# + pycharm={"name": "#%%\n"} id="j2OIbvJTm62l" outputId="9170fed3-adf5-4aab-e724-f6e342249732" colab={"base_uri": "https://localhost:8080/", "height": 281}
training_features_outliers_removed = training_features_outliers_marked.dropna()
training_features_outliers_removed.boxplot(figsize=(18,7))
plt.xticks([1], [''])
print("Number of rows left", training_features_outliers_removed.shape[0])


# + [markdown] pycharm={"name": "#%% md\n"} id="r8fsvlJZm62l"
# The boxplot now looks better, except for the second part (columns 60 to 120), which is `positions_std_i`.
# We can also see that if we remove all the rows with at least one detected outlier, we are left with less than half of the original data! This is due to the large number of features.
#
# We need another method for this dataset, let's instead cap the outliers to 3 sigma.

# + pycharm={"name": "#%%\n"} id="-artO4WUm62l"
def pipeline_outliers(df, std_cap=3):
  df = df.copy()

  for column in df.columns:

    mean = df[column].mean(skipna = True)
    std = df[column].std(skipna = True)
    
    df[column] = np.clip(df[column], -(mean + std_cap*std), mean + std_cap*std)

  return df


# + pycharm={"name": "#%%\n"} id="oDGrLk2Jm62l" outputId="3145fc38-ed0e-4bd3-f389-77c2a98ec641" colab={"base_uri": "https://localhost:8080/", "height": 337}
df = pipeline_outliers(training_features)
df.boxplot(figsize=(18,7))
plt.xticks([1], [''])


# + [markdown] pycharm={"name": "#%% md\n"} id="QHeOA5iqm62g"
# ## Missing data

# + [markdown] pycharm={"name": "#%% md\n"} id="T-HJFZs8m62g"
# As we saw above, there are 6 columns that have missing values (3 or 4 missing values each). Many classifiers do not handle missing values directly, such as Logistic Regression and SVM, for example. As such we need to find a way to manage the missing values.

# + [markdown] pycharm={"name": "#%% md\n"} id="ENEICm2Om62g"
# There are many different ways of handling missing values and we will explore a few of them here. To get started, let's examine the features/columns that contain missing data. The two visualizations chosen for each of the features/columns are:
#
# - *Histogram* - This will give a good indication of the distribution, for example if it appears to be normal.
# - *Boxplot* - We get some additional information from the boxplot showing the median, quartiles as well as outliers.
#
# Let's plot the distributions for the columns with missing data:

# + pycharm={"name": "#%%\n"} id="3-dLLDaPm62g" outputId="66d2de3f-72c3-4e08-80cc-832bca38404b" colab={"base_uri": "https://localhost:8080/", "height": 332}
def plot_distributions_for_columns(dataframe, columns):
    # Plot distributions for each of the columns that have missing values
    figure, axes = plt.subplots(2, len(columns), figsize=(12, 6))

    for index, column in enumerate(columns):
        # plot a histogram of the column for the first row
        dataframe[column].plot(
            kind='hist', ax=axes[0, index], title=column, bins=15
        )
        # Do a box plot as well
        sns.boxplot(y=dataframe[column], ax=axes[1, index]).set_title(column)

    plt.tight_layout()


plot_distributions_for_columns(training_features, columns_with_nulls.index)

# + [markdown] pycharm={"name": "#%% md\n"} id="TBDoZnucm62h"
# The distributions are quite different, which mean we may need different imputation techniques for each column.
# For example, the columns `positions_mean_9` and `positions_mean_15` are clearly not normal distributions and replacing missing values with the mean would likely not be ideal.
# For instance in the case of `positions_mean_9` the mean is close to 0 where few other samples are, moreover this feature may even need to be split into two separate features as it appears to be the combination of two gaussian distributions.
#
# Also, we do not know if the missing values themselves have a significance, i.e. we might want to create a separate column to indicate that a missing value is present or not. There are relatively few rows that have missing values, though, that may limit the usefulness of this technique and usefulness will also depend on the classifier used in the end.
#
# For now, we will implement support for the following imputation strategies:
# - Replacing with **mean**
# - **Drop the rows** containing at least one missing value
# - **K-Nearest Neighbour (KNN)** imputation, i.e. use the mean value of the K nearest neighbours

# + pycharm={"name": "#%%\n"} id="AjTqPq0Qm62h"
from sklearn.impute import KNNImputer, SimpleImputer

def impute(dataframe, imputer_class, **kwargs):
    imputer = imputer_class(**kwargs)
    dataframe = dataframe.copy()
    dataframe[dataframe.columns] = imputer.fit_transform(dataframe.values)
    return dataframe

# KNN imputation
def impute_knn(dataframe):
    return impute(dataframe, KNNImputer, n_neighbors=2, weights='uniform')

# Drop rows
def impute_drop_rows(dataframe):
    return dataframe.dropna()

# Mean imputation
def impute_mean(dataframe):
    return impute(
        dataframe, SimpleImputer, missing_values=np.nan, strategy='mean'
    )



# + [markdown] pycharm={"name": "#%% md\n"} id="o09ETbgLm62h"
# Let's compare the mean and KNN imputation methods.

# + pycharm={"name": "#%%\n"} id="C8MiBZ8Jm62h"
columns_null_sum = training_features.isnull().sum()
columns_with_nulls = columns_null_sum[columns_null_sum > 0]

# + pycharm={"name": "#%%\n"} id="PUSYg7HNm62h"
#training_features = training_data[training_feature_columns]
all_na_values = training_features.isna()

# + [markdown] pycharm={"name": "#%% md\n"} id="FTGyuLvum62h"
# First let's execute both imputations individually.
# Starting with the mean imputation, the imputed values are the following:

# + pycharm={"name": "#%%\n"} id="v14LUk-Km62i" outputId="3a6b15cc-155c-4759-bd5f-40c0ce506048" colab={"base_uri": "https://localhost:8080/"}
training_data_mean_imputed = impute_mean(training_features)
training_data_mean_imputed.values[all_na_values]

# + [markdown] pycharm={"name": "#%% md\n"} id="lGKhqpOsm62i"
# While the KNN-imputed values look like this:

# + pycharm={"name": "#%%\n"} id="FscTkmDLm62i" outputId="c888a8ae-11c4-450b-d89f-b553060a65ea" colab={"base_uri": "https://localhost:8080/"}
training_data_knn_imputed = impute_knn(training_features)
training_data_knn_imputed.values[all_na_values]

# + [markdown] pycharm={"name": "#%% md\n"} id="EtCVnW3um62i"
# ### Comparison of mean and KNN
# Now it is time to compare the results and visualize the differences on a histogram

# + pycharm={"name": "#%%\n"} id="Ne7NYuHXm62i" outputId="53fa39df-ee95-4aac-a6d6-c46eef07f36a" colab={"base_uri": "https://localhost:8080/", "height": 265}
differences = (
    training_data_mean_imputed.values[all_na_values]
    - training_data_knn_imputed.values[all_na_values]
)
plt.hist(differences);

# + [markdown] pycharm={"name": "#%% md\n"} id="rQTrrd-qm62i"
# We can conclude that the differences between the methods are not significant, the mode is close to zero, for most of the missing values the two methods give very similar imputed values - more than half of the values are in the range [-0.1,0.1]
#
# Now let's look at how the difference is distributed for each column using a boxplot and a swarm plot.
#

# + pycharm={"name": "#%%\n"} id="WGeg5UJDm62j" outputId="7b4edc51-7010-4610-f78f-8f5fcdde7b0b" colab={"base_uri": "https://localhost:8080/", "height": 297}
na_rows, na_columns = np.where(all_na_values)
dataframe_differences_columns = pd.DataFrame(
    {'diff': differences, 'column': training_features.columns[na_columns]}
)

fig, axs = plt.subplots(ncols=2)

axis = sns.boxplot(
    x='column', data=dataframe_differences_columns, y='diff', ax=axs[0]
)
axis.set_xticklabels(axis.get_xticklabels(), rotation=90);

axis = sns.swarmplot(
    x='column', data=dataframe_differences_columns, y='diff', ax=axs[1]
)
axis.set_xticklabels(axis.get_xticklabels(), rotation=90);

fig.tight_layout()


# + [markdown] pycharm={"name": "#%% md\n"} id="BeH5QKZUm62j"
# When comparing the imputed values between the 'mean' and 'KNN' approaches, we see that they produce very similar values for columns `positions_mean_7` and `positions_mean_16` but larger differences by varying degrees for the other columns. The largest (absolute) deviance is for `positions_mean_9`.
#
#
# Based on this it is likely that we may need to employ different imputation techniques depending on the column/feature.

# + [markdown] pycharm={"name": "#%% md\n"} id="7EvO3QTJm62j"
# For now, however, let's use KNN imputation and make sure that after the imputation there are no more missing values in our dataset. We will bring these three different methods into Task 2 of the project.
#

# + id="cJCnEdNqok3O" pycharm={"name": "#%%\n"}
# Let's create a pipeline function that uses the KNN imputation
def pipeline_missing_values(df):
  df = df.copy()
  
  columns_null_sum = df.isnull().sum()
  columns_with_nulls = columns_null_sum[columns_null_sum > 0]

  df[columns_with_nulls.index] = impute_knn(df[columns_with_nulls.index])
  assert df.isnull().sum().sum() == 0

  return df


# + [markdown] pycharm={"name": "#%% md\n"} id="UIZibV1-m62j"
# ## Scaling
# First we will take a look at how the data looks by feature/column. While the dataset contains a lot of features, we can use a boxplot to get an overview understanding of how the different columns compare.

# + [markdown] pycharm={"name": "#%% md\n"} id="1UU_s9ajm62j"
# Let's make a boxplot for every feature to get an overview of how they all relate in terms of range & centre. No need to have labels for the feature names, we just want to show all of them in one plot.

# + pycharm={"name": "#%%\n"} id="zZ8mc_GWm62k" outputId="2af47191-5b38-4fc8-f431-1002b2a0cc98" colab={"base_uri": "https://localhost:8080/", "height": 337}
training_features.boxplot(figsize=(18, 7))
plt.xticks([1], [''])


# + [markdown] pycharm={"name": "#%% md\n"} id="IabX4HxSm62k"
# The columns are clearly not all scaled to the same range and they have different means. As such a scaler that centers the mean and normalizes (scale to the variance) may be suitable. Let's use scikit's `StandardScaler` for this.
#
# The `StandardScaler`normalizes the data so that the mean becomes zero, and the variance one, i.e. the scaled dataset follows a *standard* normal distribution.

# + pycharm={"name": "#%%\n"} id="aeXO-9R9m62k" outputId="3e3b1dda-14ed-4bb7-f572-190ac361ba85" colab={"base_uri": "https://localhost:8080/", "height": 300}
def pipeline_scale(dataframe):
    scaler = preprocessing.StandardScaler()
    scaled_values = scaler.fit_transform(dataframe.values)
    return pd.DataFrame(scaled_values)


training_features_scaled = pipeline_scale(training_features)
training_features_scaled.head()

# + [markdown] pycharm={"name": "#%% md\n"} id="z3ewuEEFm62k"
# Looks good, i.e. the mean is about 0 and the standard deviation is around 1. All the columns have now been scaled. Let's rerun the boxplot.

# + colab={"base_uri": "https://localhost:8080/", "height": 335} id="W1uf0D3QsqPH" outputId="8c9271bc-c2aa-4a2c-e1fd-10d410f6655d" pycharm={"name": "#%%\n"}
training_features_scaled.boxplot(figsize=(18, 7))
plt.xticks([1], [''])

# + [markdown] id="Unie-y19uJpv" pycharm={"name": "#%% md\n"}
# # Pipeline

# + [markdown] id="wTBLe3qVvs2f" pycharm={"name": "#%% md\n"}
# In the Methods section above we have defined the pipeline methods needed for handling outliers, missing data and scaling. Let's put them all together.

# + colab={"base_uri": "https://localhost:8080/", "height": 337} id="1kLsdpktv7pS" outputId="ca6ad9cb-0932-4d61-d04c-84b137dc5174" pycharm={"name": "#%%\n"}
# Load the data
training_features, training_labels = load_data()

# Plot before pipeline
training_features.boxplot(figsize=(18, 7))
plt.xticks([1], [''])

# + [markdown] id="NqTkoxZQv7EW" pycharm={"name": "#%% md\n"}
# Run the pipeline and rerun the boxplot on the resulting dataset

# + colab={"base_uri": "https://localhost:8080/", "height": 281} id="8TDqvcNTzgQ4" outputId="9b00742b-403e-41bd-a19f-badf2f8cffb3" pycharm={"name": "#%%\n"}
df = pipeline_outliers(training_features, std_cap=3)
df = pipeline_missing_values(df)
df = pipeline_scale(df)

df.boxplot(figsize=(18, 7))
plt.xticks([1], [''])
