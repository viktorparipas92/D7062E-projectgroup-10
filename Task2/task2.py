# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# + [markdown] id="xsm10C7KRTiq" pycharm={"name": "#%% md\n"}
# #Task 1

# + [markdown] id="sNw45BR0m62X" pycharm={"name": "#%% md\n"}
# ## Importing dependencies
# We used the following libraries in the first task:
# - `pandas` to manipulate the data
# - `scikit-learn` for imputing and scaling the data
# - `seaborn` and `matplotlib` for visualization.

# + id="4CRS48qom62Z" pycharm={"name": "#%%\n"}
import pandas as pd
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


# + [markdown] id="XyY6wwS-m62a" pycharm={"name": "#%% md\n"}
# ## Data exploration
# Let's load the training dataset from the corresponding .csv file.
# Since we know that the columns represent the mean/standard deviation of the positions and angles of the 60 points, respectively, followed by the label name and code, let us rename the columns accordingly to allow for easier reading.

# + id="KNoDyk_fm62a" pycharm={"name": "#%%\n"}
def load_data():
  training_data = pd.read_csv('train-final.csv', header=None)
  test_data = pd.read_csv('test-final.csv', header=None)

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
  training_codes = training_data.label_code

  test_data.rename(name_mappings, axis=1, inplace=True)
  test_feature_columns = test_data.columns[:-2]

  test_features = test_data[test_feature_columns]
  test_labels = test_data.label_name

  return training_features, training_labels, training_codes, test_features, test_labels


training_features, training_labels, training_codes, test_features, test_labels = load_data()

# + [markdown] id="_aYlx2ckm62b" pycharm={"name": "#%% md\n"}
# Let's show some of the data

# + colab={"base_uri": "https://localhost:8080/", "height": 236} id="paURPIp3m62b" outputId="142e6a7e-3558-4685-c990-c24ddfbeee78" pycharm={"name": "#%%\n"}
training_features.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 236} id="Yir2V2VRJNj8" outputId="5c6b91bf-f558-48cc-8b48-e598a2a7b000" pycharm={"name": "#%%\n"}
test_features.head()

# + [markdown] id="DYPTNtFmm62c" pycharm={"name": "#%% md\n"}
# How many different labels do we have in the training dataset?

# + colab={"base_uri": "https://localhost:8080/"} id="l8hDw1bgm62c" outputId="bb0922e6-7595-45a2-8521-30a573391ad6" pycharm={"name": "#%%\n"}
number_of_classes = training_labels.nunique()
number_of_classes

# + [markdown] id="2NwxuXvqm62c" pycharm={"name": "#%% md\n"}
# Now let's take a look at how many occurrences we have of each label.

# + colab={"base_uri": "https://localhost:8080/", "height": 321} id="_O-CkrFfm62d" outputId="fd23a889-cae4-4877-b2f1-64e5855fcb5a" pycharm={"name": "#%%\n"}
training_labels.value_counts().plot(kind='bar', figsize=(10, 4))

# + [markdown] id="YlISrYJpm62d" pycharm={"name": "#%% md\n"}
# We can see that `child` is the most common label in the training dataset and that `go` is the least common label.

# + [markdown] id="AhOTmkwRm62d" pycharm={"name": "#%% md\n"}
# Now let's look for columns that have missing values. The missing values are in the following columns (along with the missing value count):
#

# + colab={"base_uri": "https://localhost:8080/"} id="rTfhsEE8m62f" outputId="a6ad1ea6-9061-437c-c399-3355bdcbf530" pycharm={"name": "#%%\n"}
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


# + [markdown] id="yprFNiVkm62j" pycharm={"name": "#%% md\n"}
# Some classifiers are more sensitive to the range, mean & outliers of the features, such as linear regression models, for example.
# In order to be able to train a wide range of classifiers and compare them, we will need to preprocess the data for scaling and outlier treatment.

# + [markdown] id="t9yB47dLm62k" pycharm={"name": "#%% md\n"}
# Let's see if the dataset also contains outliers. There are quite a few way to detect outliers (Source:
# [Outlier detection methods in Scikit-Learn](https://scikit-learn.org/stable/modules/outlier_detection.html)):
# - Isolation forest
# - Local outlier factor
# - One-class support vector machine (SVM)
# - Elliptic envelope
#
# We start by doing a boxplot for all features to get a visual indication of the outlier situation.

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="r77J-ZiOm62k" outputId="85265cc2-eeff-4995-bfc8-510d5910c20e" pycharm={"name": "#%%\n"}
training_features.boxplot(figsize=(18,7))
plt.xticks([1], [''])

# + [markdown] id="m4tCX_XRnYRS" pycharm={"name": "#%% md\n"}
# Based on the boxplot, there appears to be many columns with outliers. Many classifiers, e.g. linear classifiers like Logistic Regression will not handle outliers well, so we need to find a way to handle also outliers.

# + [markdown] id="J1sVekRXoT4Q" pycharm={"name": "#%% md\n"}
# ## Methods

# + [markdown] id="Kq-ne6mOolOy" pycharm={"name": "#%% md\n"}
# ### Outliers
#
# As we saw in the boxplot above, there are many columns with outliers. And while there are many methods to detect outliers,let's begin with just identifying the **values** that are farthest from the mean. 
#
# A simple approach is to identify the values that lie outside of 3$\sigma$ (that is, three times the standard deviation) as outliers, and drop the rows that have at least one outlier. Let's give it a try.

# + id="sOaxwVV_m62k" pycharm={"name": "#%%\n"}
#training_features_outliers_marked = training_features[abs(training_features) <= 3]
from scipy import stats

training_features_outliers_marked = training_features[
    np.abs(stats.zscore(training_features.fillna(training_features.mean()))) < 3
]

# + colab={"base_uri": "https://localhost:8080/", "height": 236} id="gGFZoxbwpqaE" outputId="0f86cb71-63cb-4585-9128-65fa55ae8b62" pycharm={"name": "#%%\n"}
training_features_outliers_marked.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="j2OIbvJTm62l" outputId="1f94e750-c063-4925-c356-c84e5809e4b0" pycharm={"name": "#%%\n"}
training_features_outliers_removed = training_features_outliers_marked.dropna()
training_features_outliers_removed.boxplot(figsize=(18,7))
plt.xticks([1], [''])
print("Number of rows left", training_features_outliers_removed.shape[0])


# + [markdown] id="r8fsvlJZm62l" pycharm={"name": "#%% md\n"}
# The boxplot now looks better, except for the second part (columns 60 to 120), which is `positions_std_i`.
# We can also see that if we remove all the rows with at least one detected outlier, we are left with less than half of the original data! This is due to the large number of features.
#
# We need another method for this dataset, let's instead cap the outliers to 3 sigma.

# + id="-artO4WUm62l" pycharm={"name": "#%%\n"}
def pipeline_outliers(df, std_cap=3):
  df = df.copy()

  for column in df.columns:

    mean = df[column].mean(skipna = True)
    std = df[column].std(skipna = True)
    
    df[column] = np.clip(df[column], -(mean + std_cap*std), mean + std_cap*std)

  return df


# + colab={"base_uri": "https://localhost:8080/", "height": 436} id="oDGrLk2Jm62l" outputId="e17f2c4d-80f8-46d8-cc5f-49c8c48a0a4c" pycharm={"name": "#%%\n"}
df = pipeline_outliers(training_features)
df.boxplot(figsize=(18,7))
plt.xticks([1], [''])


# + [markdown] id="QHeOA5iqm62g" pycharm={"name": "#%% md\n"}
# ### Missing data

# + [markdown] id="T-HJFZs8m62g" pycharm={"name": "#%% md\n"}
# As we saw above, there are 6 columns that have missing values (3 or 4 missing values each). Many classifiers do not handle missing values directly, such as Logistic Regression and SVM, for example. As such we need to find a way to manage the missing values.

# + [markdown] id="ENEICm2Om62g" pycharm={"name": "#%% md\n"}
# There are many different ways of handling missing values and we will explore a few of them here. To get started, let's examine the features/columns that contain missing data. The two visualizations chosen for each of the features/columns are:
#
# - *Histogram* - This will give a good indication of the distribution, for example if it appears to be normal.
# - *Boxplot* - We get some additional information from the boxplot showing the median, quartiles as well as outliers.
#
# Let's plot the distributions for the columns with missing data:

# + colab={"base_uri": "https://localhost:8080/", "height": 441} id="3-dLLDaPm62g" outputId="4bfb4a02-a8ef-43b4-b3c2-443fe6ee8906" pycharm={"name": "#%%\n"}
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

# + [markdown] id="TBDoZnucm62h" pycharm={"name": "#%% md\n"}
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

# + id="AjTqPq0Qm62h" pycharm={"name": "#%%\n"}
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



# + [markdown] id="o09ETbgLm62h" pycharm={"name": "#%% md\n"}
# Let's compare the mean and KNN imputation methods.

# + id="C8MiBZ8Jm62h" pycharm={"name": "#%%\n"}
columns_null_sum = training_features.isnull().sum()
columns_with_nulls = columns_null_sum[columns_null_sum > 0]

# + id="PUSYg7HNm62h" pycharm={"name": "#%%\n"}
#training_features = training_data[training_feature_columns]
all_na_values = training_features.isna()

# + [markdown] id="FTGyuLvum62h" pycharm={"name": "#%% md\n"}
# First let's execute both imputations individually.
# Starting with the mean imputation, the imputed values are the following:

# + colab={"base_uri": "https://localhost:8080/"} id="v14LUk-Km62i" outputId="2a69477c-695b-4bdc-d679-30fbaa22caa3" pycharm={"name": "#%%\n"}
training_data_mean_imputed = impute_mean(training_features)
training_data_mean_imputed.values[all_na_values]

# + [markdown] id="lGKhqpOsm62i" pycharm={"name": "#%% md\n"}
# While the KNN-imputed values look like this:

# + colab={"base_uri": "https://localhost:8080/"} id="FscTkmDLm62i" outputId="d049d58e-2121-4889-dd75-6b3477d06f6f" pycharm={"name": "#%%\n"}
training_data_knn_imputed = impute_knn(training_features)
training_data_knn_imputed.values[all_na_values]

# + [markdown] id="EtCVnW3um62i" pycharm={"name": "#%% md\n"}
# ### Comparison of mean and KNN
# Now it is time to compare the results and visualize the differences on a histogram

# + colab={"base_uri": "https://localhost:8080/", "height": 265} id="Ne7NYuHXm62i" outputId="948ed5da-41e2-47f3-ddc2-55d1a5034dc1" pycharm={"name": "#%%\n"}
differences = (
    training_data_mean_imputed.values[all_na_values]
    - training_data_knn_imputed.values[all_na_values]
)
plt.hist(differences);

# + [markdown] id="rQTrrd-qm62i" pycharm={"name": "#%% md\n"}
# We can conclude that the differences between the methods are not significant, the mode is close to zero, for most of the missing values the two methods give very similar imputed values - more than half of the values are in the range [-0.1,0.1]
#
# Now let's look at how the difference is distributed for each column using a boxplot and a swarm plot.
#

# + colab={"base_uri": "https://localhost:8080/", "height": 297} id="WGeg5UJDm62j" outputId="2913062b-337f-43c9-d65a-37a0d6a1534f" pycharm={"name": "#%%\n"}
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


# + [markdown] id="BeH5QKZUm62j" pycharm={"name": "#%% md\n"}
# When comparing the imputed values between the 'mean' and 'KNN' approaches, we see that they produce very similar values for columns `positions_mean_7` and `positions_mean_16` but larger differences by varying degrees for the other columns. The largest (absolute) deviance is for `positions_mean_9`.
#
#
# Based on this it is likely that we may need to employ different imputation techniques depending on the column/feature.

# + [markdown] id="7EvO3QTJm62j" pycharm={"name": "#%% md\n"}
# For now, however, let's use KNN imputation and make sure that after the imputation there are no more missing values in our dataset. We will bring these three different methods into Task 2 of the project.
#

# + id="cJCnEdNqok3O" pycharm={"name": "#%%\n"}
# Let's create a pipeline function that uses the KNN imputation
def pipeline_missing_values(df, method = "knn"):
  df = df.copy()
  
  columns_null_sum = df.isnull().sum()
  columns_with_nulls = columns_null_sum[columns_null_sum > 0]

  if method == "knn":
    df[columns_with_nulls.index] = impute_knn(df[columns_with_nulls.index])
  elif method == "mean":
    df[columns_with_nulls.index] = impute_mean(df[columns_with_nulls.index])
  else:
    raise f"Unknown method {method}"

  assert df.isnull().sum().sum() == 0

  return df


# + [markdown] id="UIZibV1-m62j" pycharm={"name": "#%% md\n"}
# ### Scaling
# First we will take a look at how the data looks by feature/column. While the dataset contains a lot of features, we can use a boxplot to get an overview understanding of how the different columns compare.

# + [markdown] id="1UU_s9ajm62j" pycharm={"name": "#%% md\n"}
# Let's make a boxplot for every feature to get an overview of how they all relate in terms of range & centre. No need to have labels for the feature names, we just want to show all of them in one plot.

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="zZ8mc_GWm62k" outputId="48fe2e46-b7bc-4db0-db2e-cbbfa8989441" pycharm={"name": "#%%\n"}
training_features.boxplot(figsize=(18, 7))
plt.xticks([1], [''])


# + [markdown] id="IabX4HxSm62k" pycharm={"name": "#%% md\n"}
# The columns are clearly not all scaled to the same range and they have different means. As such a scaler that centers the mean and normalizes (scale to the variance) may be suitable. Let's use scikit's `StandardScaler` for this.
#
# The `StandardScaler`normalizes the data so that the mean becomes zero, and the variance one, i.e. the scaled dataset follows a *standard* normal distribution.

# + colab={"base_uri": "https://localhost:8080/", "height": 236} id="aeXO-9R9m62k" outputId="1c1974ff-cf0d-4c44-f5e8-9d45ec60d702" pycharm={"name": "#%%\n"}
def pipeline_scale(dataframe):
    scaler = preprocessing.StandardScaler()
    scaled_values = scaler.fit_transform(dataframe.values)
    return pd.DataFrame(scaled_values)


training_features_scaled = pipeline_scale(training_features)
training_features_scaled.head()

# + [markdown] id="z3ewuEEFm62k" pycharm={"name": "#%% md\n"}
# Looks good, i.e. the mean is about 0 and the standard deviation is around 1. All the columns have now been scaled. Let's rerun the boxplot.

# + colab={"base_uri": "https://localhost:8080/", "height": 432} id="W1uf0D3QsqPH" outputId="3e37475c-a847-40b2-853e-0703236c7b25" pycharm={"name": "#%%\n"}
training_features_scaled.boxplot(figsize=(18, 7))
plt.xticks([1], [''])

# + [markdown] id="Unie-y19uJpv" pycharm={"name": "#%% md\n"}
# ## Pipeline

# + [markdown] id="wTBLe3qVvs2f" pycharm={"name": "#%% md\n"}
# In the Methods section above we have defined the pipeline methods needed for handling outliers, missing data and scaling. Let's put them all together.

# + colab={"base_uri": "https://localhost:8080/", "height": 435} id="1kLsdpktv7pS" outputId="cd5c6199-608b-4c5e-87a2-d093207df08d" pycharm={"name": "#%%\n"}
# Load the data
training_features, training_labels, training_codes, test_features, test_labels = load_data()

# Plot before pipeline
training_features.boxplot(figsize=(18, 7))
plt.xticks([1], [''])

# + colab={"base_uri": "https://localhost:8080/", "height": 436} id="ur_pyuFHJ5ZA" outputId="63a84896-33f2-4316-f549-6066942c24dd" pycharm={"name": "#%%\n"}
test_features.boxplot(figsize=(18, 7))
plt.xticks([1], [''])


# + [markdown] id="NqTkoxZQv7EW" pycharm={"name": "#%% md\n"}
# Run the pipeline and rerun the boxplot on the resulting dataset

# + colab={"base_uri": "https://localhost:8080/", "height": 432} id="8TDqvcNTzgQ4" outputId="92498ebf-c59c-494b-d45d-76a5b69c2d34" pycharm={"name": "#%%\n"}
def run_data_pipeline(features, std_cap = 3, impute_method = "knn"):
  df = pipeline_outliers(features, std_cap=std_cap)
  df = pipeline_missing_values(df, method = impute_method)
  df = pipeline_scale(df)

  return df

train_df = run_data_pipeline(training_features)
train_df.boxplot(figsize=(18, 7))
plt.xticks([1], [''])

# + colab={"base_uri": "https://localhost:8080/", "height": 432} id="ZaGMyZWgKufn" outputId="56506dee-769b-4928-e543-abef143b2b75" pycharm={"name": "#%%\n"}
test_df = run_data_pipeline(test_features)

test_df.boxplot(figsize=(18, 7))
plt.xticks([1], [''])

# + [markdown] id="-reuHaPjRu-4" pycharm={"name": "#%% md\n"}
# #Task 2

# + [markdown] id="Iew8uiL1L0wx" pycharm={"name": "#%% md\n"}
# ## Model selection
# All avaliable classifiers from sklearn are imported along with classifications metrics for evaluation.

# + id="fJ_WBMvngTVT" pycharm={"name": "#%%\n"}
# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# import every avaliable classifier in sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

# import classifications metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


# + [markdown] id="u_f2WeK1glEx" pycharm={"name": "#%% md\n"}
#

# + [markdown] id="uWYeRvSngHzp" pycharm={"name": "#%% md\n"}
# ###Training and evaluation on the training data set

# + [markdown] id="bhMjB2hF2A7a" pycharm={"name": "#%% md\n"}
# We start by creating a function that trains and evaluates a model for each of the selected classifiers with default hyperparameters on a test/train split of the training data.

# + id="cjVriXEiLcSL" pycharm={"name": "#%%\n"}
def evaluate_all_classifiers(X_train, X_test, y_train, y_test):
    classifiers = [
        LogisticRegression(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        SVC(),
        KNeighborsClassifier(),
        GaussianNB(),
        GradientBoostingClassifier(),
        AdaBoostClassifier(),
        BaggingClassifier(),
        ExtraTreesClassifier(),
        MLPClassifier(),
        # use dummy classifier to get a baseline
        DummyClassifier(strategy="most_frequent")
    ]

    # initialize a dataframe to store the results
    df_results_all = pd.DataFrame(columns=["classifier", "train_accuracy", "test_accuracy", "train_f1", "test_f1", "train_precision", "test_precision", "train_recall", "test_recall"])

    for clf in classifiers:
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        y_train_pred = clf.predict(X_train)
        print("Training")
        print(f"Classifier: {clf.__class__.__name__}")

        # save the evaluation results in a dataframe
        df_results = pd.DataFrame({"classifier": [clf.__class__.__name__],
                                    "train_accuracy": [accuracy_score(y_train, y_train_pred)],
                                        "test_accuracy": [accuracy_score(y_test, y_test_pred)],
                                        "train_f1": [f1_score(y_train, y_train_pred, average='weighted')],
                                        "test_f1": [f1_score(y_test, y_test_pred, average='weighted')],
                                        "train_precision": [precision_score(y_train, y_train_pred, average='weighted')],
                                        "test_precision": [precision_score(y_test, y_test_pred, average='weighted')],
                                        "train_recall": [recall_score(y_train, y_train_pred, average='weighted')],
                                        "test_recall": [recall_score(y_test, y_test_pred, average='weighted')],
                                        })
        # append the results to the dataframe
        df_results_all = df_results_all.append(df_results, ignore_index=True)

    return df_results_all


# + [markdown] id="sCHy2RNGPluN" pycharm={"name": "#%% md\n"}
# The training data with our default preprocessing is split into training and test sets with an 80/20 split to avoid overfitting the models towards the final testing data.

# + colab={"base_uri": "https://localhost:8080/"} id="NbNgRaBtOY4g" outputId="2238506a-9c25-4f8b-a352-f1557239b6eb" pycharm={"name": "#%%\n"}
#import train_test_split
from sklearn.model_selection import train_test_split
X = train_df
y = training_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Traning data set shape before split: {train_df.shape}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# + [markdown] id="2DozcujgLYRf" pycharm={"name": "#%% md\n"}
# The models are then evaluated on the split data:

# + colab={"base_uri": "https://localhost:8080/"} id="YrR_WmTchOVv" outputId="619d12ef-a99f-4ee5-ebe9-c9308423aef3" pycharm={"name": "#%%\n"}
results_default = evaluate_all_classifiers(X_train, X_test, y_train, y_test)

# + colab={"base_uri": "https://localhost:8080/", "height": 426} id="w4t3IrDjhtRo" outputId="7ff05694-fdfc-49d4-b04f-8a098b364bb5" pycharm={"name": "#%%\n"}
results_default

# + [markdown] id="VfFIfRFssLjM" pycharm={"name": "#%% md\n"}
# The models are evaluated again with different preprocessing parameters (higher cap for outlier capping and mean as imputation method insetad)

# + colab={"base_uri": "https://localhost:8080/"} id="_Bxb8L7RsL68" outputId="2e4ab237-8234-4ea6-c902-7885ec89bdf3" pycharm={"name": "#%%\n"}
train_df = run_data_pipeline(training_features, std_cap = 6, impute_method="mean")
test_df = run_data_pipeline(test_features, std_cap = 6, impute_method="mean")

X = train_df
y = training_labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results_cap_mean = evaluate_all_classifiers(X_train, X_test, y_train, y_test)

# + colab={"base_uri": "https://localhost:8080/", "height": 426} id="hzHoJaUBtUHT" outputId="4f263db8-8ef9-4b92-898b-27994d2b36c1" pycharm={"name": "#%%\n"}
results_cap_mean

# + [markdown] id="Uh-Sg510x3_l" pycharm={"name": "#%% md\n"}
# At least with the default parameters for the Random Forest, we get slightly higher result on the test set by increasing the cap and changing to mean imputation.
#
# By capping we are reducing the amount of information in the dataset, so it makes sense that a classifier that can make use of the outlier information could potentially perform better (or increase risks of overfitting as well).

# + [markdown] id="rSdr57n_xb44" pycharm={"name": "#%% md\n"}
# ##Regularization

# + [markdown] id="Cc-fh8VXpLqo" pycharm={"name": "#%% md\n"}
# To reduce overfitting, the feature importance is evaluated on a `RandomForestClassifier` and reduced to 45 features in accordance with the resulting plot.

# + colab={"base_uri": "https://localhost:8080/", "height": 265} id="JDdAgEA6pLbg" outputId="3de46fc8-9b28-4811-d07e-f5eba68298d3" pycharm={"name": "#%%\n"}
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

#df_feat_imp = pd.DataFrame(zip(train_df.columns, rf.feature_importances_), columns=["feat", "feat_imp"]).sort_values("feat_imp", ascending=False).reset_index(drop=True)#
df_feat_imp = pd.DataFrame(zip(X_train.columns, rf.feature_importances_), columns=["feat", "feat_imp"]).sort_values("feat_imp", ascending=False).reset_index(drop=True)#
df_feat_imp.feat_imp.plot()

feats_to_test = df_feat_imp.loc[:45, "feat"].values.tolist()


# + [markdown] id="pKYOOm47wmij" pycharm={"name": "#%% md\n"}
# The models are then evaluated with the reduced feature set.

# + colab={"base_uri": "https://localhost:8080/"} id="OmvKbayQwjgG" outputId="1c22cff1-20d0-40f8-f1a2-5b7fae26db72" pycharm={"name": "#%%\n"}

X = train_df[feats_to_test].copy()
y = training_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results_reduced = evaluate_all_classifiers(X_train, X_test, y_train, y_test)

feats_to_test = df_feat_imp.loc[:45, "feat"].values.tolist()

# + colab={"base_uri": "https://localhost:8080/", "height": 426} id="YAQl8gPvpf-n" outputId="858a4b15-930e-466d-aee7-5f41a91f681f" pycharm={"name": "#%%\n"}
results_reduced_features

# + [markdown] id="RkgrJQCEtwUZ" pycharm={"name": "#%% md\n"}
# ##"Old" code

# + id="4AUWMg68WfT3" pycharm={"name": "#%%\n"}
test_scores_default = {}

# + [markdown] id="tgfn3gH4L_As" pycharm={"name": "#%% md\n"}
# Random forest with default hyperparameters and the default preprocessing pipeline

# + colab={"base_uri": "https://localhost:8080/"} id="EagsCfTcMC8-" outputId="4f12ea9b-ad94-4cb7-fee1-feb2a8572395" pycharm={"name": "#%%\n"}
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

train_df = run_data_pipeline(training_features)
test_df = run_data_pipeline(test_features)

rf = RandomForestClassifier(random_state=1234)
rf.fit(train_df, training_labels)
test_scores_default['RandomForest'] = rf.score(test_df, test_labels)

print("Score for Random Forest with default hyperparameters:")
print(f"Test Accuracy: {test_scores_default['RandomForest']}")
#print(test_scores['RandomForest'])

# + [markdown] id="y482KqaswFeB" pycharm={"name": "#%% md\n"}
# Still random forest, but try using different preprocessing parameters (higher cap for outlier capping and mean as imputation method insetad)

# + colab={"base_uri": "https://localhost:8080/"} id="CSoN3I4Suo6D" outputId="53f1d213-9a0a-4cf7-9dc1-c8fa37480cc7" pycharm={"name": "#%%\n"}
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

train_df = run_data_pipeline(training_features, std_cap = 6, impute_method="mean")
test_df = run_data_pipeline(test_features, std_cap = 6, impute_method="mean")

rf = RandomForestClassifier(random_state=1234) # Fix random state for reproducible results
rf.fit(train_df, training_labels)
test_scores_default['RandomForest_cap_6_mean'] = rf.score(test_df, test_labels)

print("Score for Random Forest with default hyperparameters:")
print(f"Test Accuracy: {test_scores_default['RandomForest_cap_6_mean']}")
#print(test_scores['RandomForest'])

# + [markdown] id="WZJy7TrO05IH" pycharm={"name": "#%% md\n"}
# At least with the default parameters for the Random Forest, we get slightly higher result on the test set by increasing the cap and changing to mean imputation.
#
# By capping we are reducing the amount of information in the dataset, so it makes sense that a classifier that can make use of the outlier information could potentially perform better (or increase risks of overfitting as well).

# + [markdown] id="HsyfK4_8Rdz1" pycharm={"name": "#%% md\n"}
# Optimizing hyperparameters for random forest with GridSearchCV:

# + colab={"base_uri": "https://localhost:8080/", "height": 363} id="gi7Gq7ueMczx" outputId="1d532ef1-a0c4-41bf-c430-a553bb1de001" pycharm={"name": "#%%\n"}
'''
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 80, stop = 120, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [8, 16, 32]
# Minimum number of samples required to split a node
min_samples_split = [8, 16, 32]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

train_df = run_data_pipeline(training_features)
test_df = run_data_pipeline(test_features)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 5 fold cross validation, 
# search across 360 different combinations, and use all available cores
rf_grid =  GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, verbose=2, n_jobs = 4)
# Fit the random search model
rf_grid.fit(train_df, training_labels)
'''

# + id="pQn1SHqfUNDm" pycharm={"name": "#%%\n"}
rf_grid.score(test_df, test_labels)
#rf_grid.best_params_

# + id="xAfVVNPXNBw_" pycharm={"name": "#%%\n"}
'''
{'bootstrap': False,
 'max_depth': 16,
 'max_features': 'sqrt',
 'min_samples_leaf': 2,
 'min_samples_split': 8,
 'n_estimators': 90}

score = 0.9055555555555556
'''

# + [markdown] id="7A_MttPXQlTf" pycharm={"name": "#%% md\n"}
# Decision tree with default parameters:

# + id="nnPW5r0HQtgD" pycharm={"name": "#%%\n"}
from sklearn import tree

decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(train_df, training_labels)
test_scores_default['tree'] = decision_tree.score(test_df, test_labels)

print("Score for Decision Tree with default hyperparameters:")
print(f"Test Accuracy: {test_scores_default['tree']}")

# + [markdown] id="EAO4wYe7UuQc" pycharm={"name": "#%% md\n"}
# MLP with default hyperparameters:

# + id="NbBE9yBtUvOr" pycharm={"name": "#%%\n"}
from sklearn.neural_network import MLPClassifier

MLP = MLPClassifier()
MLP.fit(train_df, training_labels)
test_scores_default['MLP'] = MLP.score(test_df, test_labels)

print("Score for MLP with default hyperparameters:")
print(f"Test Accuracy: {test_scores_default['MLP']}")

# + [markdown] id="hWItiQLkV7_H" pycharm={"name": "#%% md\n"}
# SVM with default hyperparameters:

# + id="XUP2ZzsBV_hZ" pycharm={"name": "#%%\n"}
from sklearn import svm

svm_model = svm.SVC(kernel = 'linear',gamma = 'scale', shrinking = False,)
svm_model.fit(train_df, training_labels)
test_scores_default['SVM'] = svm_model.score(test_df, test_labels)

print("Score for SVM with default hyperparameters:")
print(f"Test Accuracy: {test_scores_default['SVM']}")

# + [markdown] id="NbgiQw7bXWTg" pycharm={"name": "#%% md\n"}
# kNN with default hyperparameters:

# + id="8glEVEIjXY3M" pycharm={"name": "#%%\n"}
from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors=3)
kNN.fit(train_df, training_labels)
test_scores_default['kNN'] = kNN.score(test_df, test_labels)

print("Score for kNN with default hyperparameters:")
print(f"Test Accuracy: {test_scores_default['kNN']}")

# + id="F9ynhyLWXpq-" pycharm={"name": "#%%\n"}
print("All scores with default hyperparameters:")
print(test_scores_default)

# + id="6V8jJ0qUBlMQ" pycharm={"name": "#%%\n"}
# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# train a model with every classifier in sklearn and evaluate on both train and test set
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# import dummy classifier
from sklearn.dummy import DummyClassifier

# import classifications metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

#import train_test_split
from sklearn.model_selection import train_test_split
X = train_df
y = training_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifiers = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    SVC(),
    KNeighborsClassifier(),
    GaussianNB(),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    BaggingClassifier(),
    ExtraTreesClassifier(),
    MLPClassifier(),
    # use dummy classifier to get a baseline
    DummyClassifier(strategy="most_frequent")
]

# initialize a dataframe to store the results
df_results_all = pd.DataFrame(columns=["classifier", "train_accuracy", "test_accuracy", "train_f1", "test_f1", "train_precision", "test_precision", "train_recall", "test_recall"])

for clf in classifiers:
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)
    print("Training")
    print(f"Classifier: {clf.__class__.__name__}")

    # save the evaluation results in a dataframe
    df_results = pd.DataFrame({"classifier": [clf.__class__.__name__],
                                 "train_accuracy": [accuracy_score(y_train, y_train_pred)],
                                    "test_accuracy": [accuracy_score(y_test, y_test_pred)],
                                    "train_f1": [f1_score(y_train, y_train_pred, average='weighted')],
                                    "test_f1": [f1_score(y_test, y_test_pred, average='weighted')],
                                    "train_precision": [precision_score(y_train, y_train_pred, average='weighted')],
                                    "test_precision": [precision_score(y_test, y_test_pred, average='weighted')],
                                    "train_recall": [recall_score(y_train, y_train_pred, average='weighted')],
                                    "test_recall": [recall_score(y_test, y_test_pred, average='weighted')],
                                    })
    # append the results to the dataframe
    df_results_all = df_results_all.append(df_results, ignore_index=True)

df_results_all

# + colab={"base_uri": "https://localhost:8080/"} id="0z5bd_1ECtf0" outputId="6b98ee27-bc15-4160-a80b-b1243ba6e2ce" pycharm={"name": "#%%\n"}
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# + colab={"base_uri": "https://localhost:8080/", "height": 282} id="bB4uqvJdFrff" outputId="e26c11bc-9ffe-4171-fbe0-2ab68b8972a3" pycharm={"name": "#%%\n"}
df_feat_imp = pd.DataFrame(zip(train_df.columns, rf.feature_importances_), columns=["feat", "feat_imp"]).sort_values("feat_imp", ascending=False).reset_index(drop=True)#
df_feat_imp.feat_imp.plot()

# + id="ZJH44WiuG1v9" pycharm={"name": "#%%\n"}
feats_to_test = df_feat_imp.loc[:45, "feat"].values.tolist()

X = train_df[feats_to_test].copy()
y = training_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifiers = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    SVC(),
    KNeighborsClassifier(),
    GaussianNB(),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    BaggingClassifier(),
    ExtraTreesClassifier(),
    MLPClassifier(),
    # use dummy classifier to get a baseline
    DummyClassifier(strategy="most_frequent")
]

# initialize a dataframe to store the results
df_results_all = pd.DataFrame(columns=["classifier", "train_accuracy", "test_accuracy", "train_f1", "test_f1", "train_precision", "test_precision", "train_recall", "test_recall"])

for clf in classifiers:
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)
    print("Training")
    print(f"Classifier: {clf.__class__.__name__}")

    # save the evaluation results in a dataframe
    df_results = pd.DataFrame({"classifier": [clf.__class__.__name__],
                                 "train_accuracy": [accuracy_score(y_train, y_train_pred)],
                                    "test_accuracy": [accuracy_score(y_test, y_test_pred)],
                                    "train_f1": [f1_score(y_train, y_train_pred, average='weighted')],
                                    "test_f1": [f1_score(y_test, y_test_pred, average='weighted')],
                                    "train_precision": [precision_score(y_train, y_train_pred, average='weighted')],
                                    "test_precision": [precision_score(y_test, y_test_pred, average='weighted')],
                                    "train_recall": [recall_score(y_train, y_train_pred, average='weighted')],
                                    "test_recall": [recall_score(y_test, y_test_pred, average='weighted')],
                                    })
    # append the results to the dataframe
    df_results_all = df_results_all.append(df_results, ignore_index=True)

df_results_all

# + id="Z6s1G_VWT3RS" pycharm={"name": "#%%\n"}
train_df.head()

# + id="bD_uv4DEU6hK" pycharm={"name": "#%%\n"}
training_codes.shape

# + id="njCNl3mkVVee" pycharm={"name": "#%%\n"}
df = pd.read_csv('train-final.csv', header=None)
df.head()

# + id="-UzkFJAZV1ne" pycharm={"name": "#%%\n"}
X_train.mean()

# + colab={"base_uri": "https://localhost:8080/", "height": 502} id="olRV7BryLC5Y" outputId="6fd54bd8-f3bb-43c4-f48f-ff9e13593bee" pycharm={"name": "#%%\n"}
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split


#train_df = train_df#run_data_pipeline(training_features)
#test_df = run_data_pipeline(test_features)

X = df.iloc[:, :239]
#X = train_df
#y = training_labels
#y = training_codes
y = df[241]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)
mean_values = X_train.mean()
X_train = X_train.fillna(mean_values)
X_test = X_test.fillna(mean_values)

# logistic regression
logreg = LogisticRegression(C=200, penalty="l2")
rfe_lr = RFE(logreg, n_features_to_select=10, step=10)
rfe_lr = rfe_lr.fit(X_train, y_train)
#print(rfe.support_)
#print(rfe.ranking_)
print(X_train.columns[rfe_lr.support_])

# random forest
rf = RandomForestClassifier()
rfe_rf = RFE(rf, n_features_to_select=10, step=10)
rfe_rf = rfe_rf.fit(X_train, y_train)
#print(rfe.support_)
#print(rfe.ranking_)
print(X_train.columns[rfe_rf.support_])

X_train2 = rfe_lr.transform(X_train)
X_test2 = rfe_lr.transform(X_test)
logreg.fit(X_train2, y_train)
y_train_pred_lr = logreg.predict(X_train2)
y_test_pred_lr = logreg.predict(X_test2)

X_train2 = rfe_rf.transform(X_train)
X_test2 = rfe_rf.transform(X_test)
rf.fit(X_train2, y_train)
y_train_pred_rf = rf.predict(X_train2)
y_test_pred_rf = rf.predict(X_test2)

print(f"Logistic regression score: {accuracy_score(y_train, y_train_pred_lr):.2f}")
print(f"Logistic regression score: {accuracy_score(y_test, y_test_pred_lr):.2f}")
print(f"Random forest score: {accuracy_score(y_train, y_train_pred_rf):.2f}")
print(f"Random forest score: {accuracy_score(y_test, y_test_pred_rf):.2f}")



# + id="jIPn7EzVbVjj" pycharm={"name": "#%%\n"}
logreg.coef_.shape

# + id="ZCxLd77NbArk" pycharm={"name": "#%%\n"}
pd.DataFrame(zip(list(X_train.columns[(rfe_rf.support_)]), rf.feature_importances_), columns=["feat", "feat_imp"]).sort_values("feat_imp", ascending=False)

# + colab={"base_uri": "https://localhost:8080/", "height": 70} id="JCInmgW5kOVr" outputId="0bd9b368-6986-4ce6-9161-a217f3a72803" pycharm={"name": "#%%\n"}
'''
fig, ax = plt.subplots(ncols=1)
x = np.arange(0, len(results_all["classifier"]))

ax[0].bar(x, results_all["train_accuracy"])
ax[0].set_xticks(x)
ax[0].set_xticklabels(results_all["classifier"], rotation='vertical')

ax[1].bar(x, results_all["test_accuracy"])
ax[1].set_xticks(x)
ax[1].set_xticklabels(results_all["classifier"], rotation='vertical')

diff = results_all["train_accuracy"] - results_all["test_accuracy"]
ax[0].bar(x, diff)
ax[0].set_xticks(x)
ax[0].set_xticklabels(results_all["classifier"], rotation='vertical')


diff = results_all["train_accuracy"] - results_all["test_accuracy"]
x = np.arange(0, len(results_all["classifier"]))
plt.bar(x, diff)
plt.xticks(x, results_all["classifier"], rotation='vertical')

fig.tight_layout()
plt.show()
'''
