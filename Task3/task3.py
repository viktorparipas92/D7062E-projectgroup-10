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

# + [markdown] id="VqtF_yiLtQqU" pycharm={"name": "#%% md\n"}
# # D7062E Project
# ## Project Group 10
#
# Contributors:
# - Theo HEMBÄCK
# - PARIPÁS Viktor
# - Jerker ÅBERG
# - Kristofer ÅGREN

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
# Let's load the training dataset from the corresponding `.csv` file.
# Since we know that the columns represent the mean/standard deviation of the positions and angles of the 60 points, respectively, followed by the label name and code, let us rename the columns accordingly to allow for easier reading.

# + pycharm={"name": "#%%\n"}
def load_data():
    training_data = pd.read_csv("train-final.csv", header=None)
    test_data = pd.read_csv("test-final.csv", header=None)

    name_mappings = {
        # Feature columns
        **{i: f"positions_mean_{i}" for i in range(60)},
        **{i: f"positions_std_{i}" for i in range(60, 120)},
        **{i: f"angles_mean_{i}" for i in range(120, 180)},
        **{i: f"angles_std_{i}" for i in range(180, 240)},
        # Label columns
        **{240: "label_name", 241: "label_code"},
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

    return (
        training_features,
        training_labels,
        training_codes,
        test_features,
        test_labels,
    )


(
    training_features,
    training_labels,
    training_codes,
    test_features,
    test_labels,
) = load_data()

# + [markdown] pycharm={"name": "#%% md\n"}
# Let's show some of the data

# + pycharm={"name": "#%%\n"}
training_features.head()

# + pycharm={"name": "#%%\n"}
test_features.head()

# + [markdown] pycharm={"name": "#%% md\n"}
# How many different labels do we have in the training dataset?

# + pycharm={"name": "#%%\n"}
number_of_classes = training_labels.nunique()
number_of_classes

# + [markdown] pycharm={"name": "#%% md\n"}
# Now let's take a look at how many occurrences we have of each label.

# + pycharm={"name": "#%%\n"}
training_labels.value_counts().plot(kind="bar", figsize=(10, 4))

# + [markdown] pycharm={"name": "#%% md\n"}
# We can see that `child` is the most common label in the training dataset and that `go` is the least common label.

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Preprocessing

# + [markdown] pycharm={"name": "#%% md\n"}
# Some classifiers are more sensitive to the range, mean & outliers of the features, such as linear regression models, for example.
# In order to be able to train a wide range of classifiers and compare them, we will need to preprocess the data for scaling and outlier treatment.

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Handling outliers

# + [markdown] pycharm={"name": "#%% md\n"}
# Let's see if the dataset also contains outliers. There are quite a few way to detect outliers (Source:
# [Outlier detection methods in Scikit-Learn](https://scikit-learn.org/stable/modules/outlier_detection.html)):
# - Isolation forest
# - Local outlier factor
# - One-class support vector machine (SVM)
# - Elliptic envelope
#
# We start by doing a **boxplot** for all features to get a visual indication of the outlier situation. The boxplot (or box-and-whisker plot) shows the
# - inter-quartile range as a box
# - with the median marked in the middle,
# - as well as whiskers of given length extending from the quartiles,
# - with outliers (points outside the whiskers) plotted as individual points
# for a given column.
# Because of the last bullet point, it is a good choice to assess the extent of the outlier problem.

# + pycharm={"name": "#%%\n"}
training_features.boxplot(figsize=(18, 7))
plt.xticks([1], [""])

# + [markdown] pycharm={"name": "#%% md\n"}
# Based on the boxplot, there appears to be many columns with outliers. Many classifiers, e.g. linear classifiers like Logistic Regression will not handle outliers well, so we need to find a way to handle outliers.

# + [markdown] pycharm={"name": "#%% md\n"}
# As we saw in the boxplot above, there are many columns with outliers. And while there are many methods to detect outliers,let's begin with just identifying the **values** that are farthest from the mean.
#
# A simple approach is to identify the values that lie outside of 3$\sigma$ (that is, three times the standard deviation) as outliers, and drop the rows that have at least one outlier. Let's give it a try.

# + pycharm={"name": "#%%\n"}
from scipy import stats

training_features_outliers_marked = training_features[
    np.abs(stats.zscore(training_features.fillna(training_features.mean()))) < 3
]

# + pycharm={"name": "#%%\n"}
training_features_outliers_marked.head()

# + pycharm={"name": "#%%\n"}
training_features_outliers_removed = training_features_outliers_marked.dropna()
training_features_outliers_removed.boxplot(figsize=(18, 7))
plt.xticks([1], [""])
print(f"Number of rows left: {training_features_outliers_removed.shape[0]}")


# + [markdown] pycharm={"name": "#%% md\n"}
# The boxplot now looks better, except for the second part (columns 60 to 120), which is `positions_std_i`.
# We can also see that if we remove all the rows with at least one detected outlier, we are left with less than half of the original data! This is due to the large number of features.
#
# We need another method for this dataset, let's instead cap the outliers to 3 $\sigma$.

# + pycharm={"name": "#%%\n"}
def run_outlier_handling_pipeline(dataframe, std_cap=3):
    dataframe = dataframe.copy()
    for column in dataframe.columns:
        mean = dataframe[column].mean(skipna=True)
        std = dataframe[column].std(skipna=True)

        dataframe[column] = np.clip(
            dataframe[column],
            -(mean + std_cap * std),
            mean + std_cap * std,
        )

    return dataframe


# + pycharm={"name": "#%%\n"}
outlier_handled_data = run_outlier_handling_pipeline(training_features)
outlier_handled_data.boxplot(figsize=(18, 7))
plt.xticks([1], [""])

# + [markdown] pycharm={"name": "#%% md\n"}
# The boxplot still looks more promising than the original and we managed to maintain all of our features. Now let's move on to handling the missing data.

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Missing data

# + [markdown] pycharm={"name": "#%% md\n"}
# Now let's look for columns that have missing values. The missing values are in the following columns (along with the missing value count):

# + pycharm={"name": "#%%\n"}
columns_null_sum = training_features.isnull().sum()
columns_with_nulls = columns_null_sum[columns_null_sum > 0]

print(
    "Total amount of missing values in the dataframe:",
    training_features.isnull().sum().sum(),
)
print("Missing values in the following column indexes (and missing value count):")
print(columns_with_nulls)


# + [markdown] pycharm={"name": "#%% md\n"}
# As we saw above, there are 6 columns that have missing values (3 or 4 missing values each). Many classifiers do not handle missing values directly, such as Logistic Regression and SVM, for example. As such we need to find a way to manage the missing values.

# + [markdown] pycharm={"name": "#%% md\n"}
# There are many different ways of handling missing values and we will explore a few of them here. To get started, let's examine the features/columns that contain missing data. The two visualizations chosen for each of the features/columns are:
#
# - *Histogram* - This will give a good indication of the distribution, for example if it appears to be normal.
# - *Boxplot* - We get some additional information from the boxplot showing the median, quartiles as well as outliers.
#
# Let's plot the distributions for the columns with missing data:

# + pycharm={"name": "#%%\n"}
def plot_distributions_for_columns(dataframe, columns):
    figure, axes = plt.subplots(2, len(columns), figsize=(12, 6))

    for index, column in enumerate(columns):
        dataframe[column].plot(kind="hist", ax=axes[0, index], title=column, bins=15)
        sns.boxplot(y=dataframe[column], ax=axes[1, index]).set_title(column)

    plt.tight_layout()


plot_distributions_for_columns(training_features, columns_with_nulls.index)

# + [markdown] pycharm={"name": "#%% md\n"}
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

# + pycharm={"name": "#%%\n"}
from sklearn.impute import KNNImputer, SimpleImputer


def impute(dataframe, imputer_class, **kwargs):
    imputer = imputer_class(**kwargs)
    dataframe = dataframe.copy()
    dataframe[dataframe.columns] = imputer.fit_transform(dataframe.values)
    return dataframe


def impute_knn(dataframe):
    return impute(dataframe, KNNImputer, n_neighbors=2, weights="uniform")


def impute_drop_rows(dataframe):
    return dataframe.dropna()


def impute_mean(dataframe):
    return impute(dataframe, SimpleImputer, missing_values=np.nan, strategy="mean")


# + [markdown] pycharm={"name": "#%% md\n"}
# Let's compare the mean and KNN imputation methods.

# + pycharm={"name": "#%%\n"}
all_na_values = training_features.isna()

# + [markdown] pycharm={"name": "#%% md\n"}
# First let's execute both imputations individually.
# Starting with the mean imputation, the imputed values are the following:

# + pycharm={"name": "#%%\n"}
training_data_mean_imputed = impute_mean(training_features)
training_data_mean_imputed.values[all_na_values]

# + [markdown] pycharm={"name": "#%% md\n"}
# While the KNN-imputed values look like this:

# + pycharm={"name": "#%%\n"}
training_data_knn_imputed = impute_knn(training_features)
training_data_knn_imputed.values[all_na_values]

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Comparison of mean and KNN
# Now it is time to compare the results and visualize the differences on a histogram

# + pycharm={"name": "#%%\n"}
differences = (
    training_data_mean_imputed.values[all_na_values]
    - training_data_knn_imputed.values[all_na_values]
)
plt.hist(differences)

# + [markdown] pycharm={"name": "#%% md\n"}
# We can conclude that the differences between the methods are not significant, the mode is close to zero, for most of the missing values the two methods give very similar imputed values - more than half of the values are in the range [-0.1,0.1]
#
# Now let's look at how the difference is distributed for each column using a boxplot and a swarm plot.
#
# A **swarm plot** is a categorical scatter plot with points adjusted to be non-overlapping, also a good visual representation of the distribution of the differences.
#

# + pycharm={"name": "#%%\n"}
na_rows, na_columns = np.where(all_na_values)
dataframe_differences_columns = pd.DataFrame(
    {"diff": differences, "column": training_features.columns[na_columns]}
)

fig, axs = plt.subplots(ncols=2)

axis = sns.boxplot(x="column", data=dataframe_differences_columns, y="diff", ax=axs[0])
axis.set_xticklabels(axis.get_xticklabels(), rotation=90)

axis = sns.swarmplot(
    x="column", data=dataframe_differences_columns, y="diff", ax=axs[1]
)
axis.set_xticklabels(axis.get_xticklabels(), rotation=90)

fig.tight_layout()


# + [markdown] pycharm={"name": "#%% md\n"}
# When comparing the imputed values between the *mean* and *KNN* approaches, we see that they produce very similar values for columns `positions_mean_7` and `positions_mean_16` but larger differences by varying degrees for the other columns. The largest (absolute) deviance is for `positions_mean_9`.
#
#
# Based on this it is likely that we may need to employ different imputation techniques depending on the column/feature.

# + [markdown] pycharm={"name": "#%% md\n"}
# For now, however, let's use KNN imputation and make sure that after the imputation there are no more missing values in our dataset. We will bring these three different methods into Task 2 of the project.
#

# + pycharm={"name": "#%%\n"}
def run_missing_value_pipeline(dataframe, method="knn"):
    dataframe = dataframe.copy()

    columns_null_sum = dataframe.isnull().sum()
    columns_with_nulls = columns_null_sum[columns_null_sum > 0]

    if method == "knn":
        dataframe[columns_with_nulls.index] = impute_knn(
            dataframe[columns_with_nulls.index]
        )
    elif method == "mean":
        dataframe[columns_with_nulls.index] = impute_mean(
            dataframe[columns_with_nulls.index]
        )
    else:
        raise f"Unknown method {method}"

    assert dataframe.isnull().sum().sum() == 0

    return dataframe


# + [markdown] pycharm={"name": "#%% md\n"}
# ### Scaling
# First we will take a look at how the data looks by feature/column. While the dataset contains a lot of features, we can use a boxplot to get an overview understanding of how the different columns compare.

# + [markdown] pycharm={"name": "#%% md\n"}
# Let's make a boxplot again for every feature to get an overview of how they all relate in terms of range & centre. No need to have labels for the feature names, we just want to show all of them in one plot.

# + pycharm={"name": "#%%\n"}
training_features.boxplot(figsize=(18, 7))
plt.xticks([1], [""])


# + [markdown] pycharm={"name": "#%% md\n"}
# The columns are clearly not all scaled to the same range and they have different means. As such a scaler that centers the mean and normalizes (scale to the variance) may be suitable. Let's use scikit's `StandardScaler` for this.
#
# The `StandardScaler`normalizes the data so that the mean becomes zero, and the variance one, i.e. the scaled dataset follows a *standard* normal distribution.

# + pycharm={"name": "#%%\n"}
def run_scaling_pipeline(dataframe):
    scaler = preprocessing.StandardScaler()
    scaled_values = scaler.fit_transform(dataframe.values)
    return pd.DataFrame(scaled_values)


training_features_scaled = run_scaling_pipeline(training_features)
training_features_scaled.describe()

# + [markdown] pycharm={"name": "#%% md\n"}
# Looks good, i.e. the mean is about 0 and the standard deviation is around 1. All the columns have now been scaled. Let's rerun the boxplot.

# + pycharm={"name": "#%%\n"}
training_features_scaled.boxplot(figsize=(18, 7))
plt.xticks([1], [""])


# + [markdown] pycharm={"name": "#%% md\n"}
# We can see that the range of the different columns now matches, barring the outliers that were handled in another pipeline.
#
# Let's put the three preprocessing methods all together.

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Pipeline

# + [markdown] pycharm={"name": "#%% md\n"}
# In the *Preprocessing* section above we have defined the pipeline methods needed for handling outliers, missing data and scaling. Let's put them all together.

# + [markdown] pycharm={"name": "#%% md\n"}
# Run the pipeline and re-run the boxplot on the resulting dataset

# + pycharm={"name": "#%%\n"}
def run_data_pipeline(features, std_cap=3, impute_method="knn"):
    dataframe = run_outlier_handling_pipeline(features, std_cap=std_cap)
    dataframe = run_missing_value_pipeline(dataframe, method=impute_method)
    dataframe = run_scaling_pipeline(dataframe)

    return dataframe


training_data = run_data_pipeline(training_features)
training_data.boxplot(figsize=(18, 7))
plt.xticks([1], [""])

# + pycharm={"name": "#%%\n"}
test_data = run_data_pipeline(test_features)
test_data.boxplot(figsize=(18, 7))
plt.xticks([1], [""])

# + [markdown] pycharm={"name": "#%% md\n"}
# # Task 2

# + [markdown] pycharm={"name": "#%% md\n"}
# The following observations have been made of the dataset and affects how models should be trained:
#
# - There is a low number, less than 30, of samples per class. As such, overfitting will likely be a problem with more advanced classifiers like decision trees, for example. To address this, we will use a cross-validation (CV) method to evaluate the fit. Furthermore, to ensure we have the same distribution in each fold of the CV, a stratified fold method will be used.
#   - A stratified fold means each set contains approximately the same percentage of samples of each target class as the complete set. (Source:
# [Stratified k-fold](https://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold))
# - The data set is very *wide* - that is, there are a very large amount of features (240) compared to the number of samples (540). To address this, a feature reduction will be used. There are many options for reducing features, such as removing features with strong correlation, low variance or fitting a classifier and removing those features that have low importance. Dimensionality reduction approaches are also common, such as Principal Component Analysis (PCA), for example.
# - We will compare RFE to PCA, and select the best performing option.

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Feature reduction

# + [markdown] pycharm={"name": "#%% md\n"}
# ### RFE
#
# To reduce features, we have selected a Recursive Feature Elimination method using a Support Vector Machine classifier. The reason for selecting a SVM classifier is the combination of speed and the ability to capture non-linearities in the dataset.

# + pycharm={"name": "#%%\n"}
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn import svm


training_data = run_data_pipeline(training_features, std_cap=6, impute_method="mean")


def reduce(n_feats, dataframe):

    rfe_selector = RFECV(
        estimator=svm.SVC(kernel="linear"),
        cv=StratifiedKFold(n_splits=10),
        min_features_to_select=n_feats,
        step=1,
        n_jobs=4,
    )

    rfe_selector.fit(dataframe, training_labels)

    selected_cols = dataframe.columns[rfe_selector.get_support()]
    score = np.mean(rfe_selector.cv_results_["mean_test_score"])

    print(n_feats, len(selected_cols), score)

    return selected_cols, score


# Evaluate a few different minimum features to select
rfe_results = [reduce(n, training_data) for n in [15, 45]]

# + [markdown] pycharm={"name": "#%% md\n"}
# As seen above, the cross validated performance peaks at 40 selected features, at around an accuracy of 0.851.

# + [markdown] pycharm={"name": "#%% md\n"}
# ### PCA
#
# Let's also look at how PCA would perform, also evaluating it using the same SVM classifiers as in the RFE case and same amount of CV folds (10). Code below is based on sklearn example at https://scikit-learn.org/stable/auto_examples/compose/plot_digits_pipe.html

# + pycharm={"name": "#%%\n"}
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pca = PCA()

pipe = Pipeline(steps=[("pca", pca), ("svm", svm.SVC(kernel="linear"))])
param_grid = {"pca__n_components": np.arange(10, 100, 5)}
search = GridSearchCV(pipe, param_grid, cv=10, n_jobs=4)
search.fit(training_data, training_labels)

print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

# + [markdown] pycharm={"name": "#%% md\n"}
# Using PCA, we achieve the highest accuracy of 0.804 on the same cross validated classifiers as RFE above using 85 PCA components.

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Conclusion
#
# The RFE approach gives us higher accuracy with fewer features than PCA, so we will select the RFE approach instead.

# + pycharm={"name": "#%%\n"}
rfe_best_ix = np.argmin([score for _, score in rfe_results])
rfe_columns, _ = rfe_results[rfe_best_ix]

train_reduced_df = training_data[training_data.columns[rfe_columns]]
test_reduced_df = test_data[test_data.columns[rfe_columns]]

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Model selection
# First we get a baseline score for some of the avaliable classifiers in sklearn. The classifiers are imported from sklearn along with classification metrics for evaluation. The baseline scores are calculated using cross validation with 10 folds.

# + pycharm={"name": "#%%\n"}
import warnings

warnings.filterwarnings("ignore")

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

from sklearn.model_selection import cross_validate


# + [markdown] pycharm={"name": "#%% md\n"}
# ### Training and evaluation on the training set

# + [markdown] pycharm={"name": "#%% md\n"}
# We create a function that trains and evaluates a model for each of the selected classifiers with default hyperparameters and run it on the feature-reduced training data set.

# + pycharm={"name": "#%%\n"}
def evaluate_all_classifiers(train_df):
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
        DummyClassifier(strategy="most_frequent"),
    ]

    all_results = []

    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    for clf in classifiers:

        scores = cross_validate(
            clf,
            train_df,
            training_labels,
            scoring=scoring,
            cv=10,
            return_train_score=True,
        )

        print(
            f"Classifier: {clf.__class__.__name__}",
            "test accuracy",
            np.mean(scores["test_accuracy"]),
        )

        scores_mean = {a: np.mean(scores[a]) for a in scores.keys()}
        scores_mean["classifier"] = clf.__class__.__name__

        # save the evaluation results in a dataframe
        all_results.append(pd.DataFrame([scores_mean]))

    return pd.concat(all_results)


results = evaluate_all_classifiers(train_reduced_df)

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Baseline scores

# + pycharm={"name": "#%%\n"}
results.sort_values(by=["test_accuracy"])

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Conclusion
#
# The best performing classifier (using CV with 10 folds) is the `ExtraTreeClassifier` with an accuracy of approximately 0.93.

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Fine-tuning models

# + [markdown] pycharm={"name": "#%% md\n"}
# Next step is to find optimal hyperparameters on the best performing classifier, `ExtraTreesClassifier`. We will do so by doing a CV Grid Search and will tune the following hyperparameters of the classifier:
#
# - n_estimators, controls the number of trees generated. Higher can give better performance, but may also increase likelihood of overfitting. The default value is 100 so we will probe around that value.
# - max_depth, the maximum depth of the tree. The default value is None, i.e. the max depth is not restricted so we first tested a wide range and concluded around 16 gives good results.
#

# + pycharm={"name": "#%%\n"}
param_grid = {
    "n_estimators": np.arange(95, 105, 1),
    "max_depth": [15, 16, 17, 18],
    "bootstrap": [False, True],
}

cv = StratifiedKFold(n_splits=10)

searchCV = GridSearchCV(
    estimator=ExtraTreesClassifier(random_state=42),
    scoring="accuracy",
    cv=cv,
    param_grid=param_grid,
    verbose=True,
    n_jobs=4,
)

searchCV.fit(train_reduced_df, training_labels)

# + pycharm={"name": "#%%\n"}
searchCV.best_params_, searchCV.best_score_

# + [markdown] pycharm={"name": "#%% md\n"}
# This gives the following best parameters to use in the evaluation on the test set:
#
#
# 1.   max_depth: 18
# 2.   n_estimators: 2
#
#

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Evaluation on the test set

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Choice of models to evaluate

# + [markdown] pycharm={"name": "#%% md\n"}
# We choose to evaluate the the models with the top five `test_accuracy` from the model selection section:
#
# 1.   `ExtraTreesClassifier` - 0.928
# 2.   `LogisticRegression` - 0.896
# 3.   `MLPClassifier` - 0.888
# 4.   `SVC` - 0.888
# 5.   `RandomForestClassifier` - 0.885
#
# After these five models the accuracy starts to drop off a bit.

# + [markdown] pycharm={"name": "#%% md\n"}
# ### Evaluation

# + [markdown] pycharm={"name": "#%% md\n"}
# We fit the models to the preprocessed and feature-reduced training set, run predictions on the test set run through the same preprocessing and feature reduction and use `accuracy_score` from `sklearn.metrics` to get the final accuracy scores.

# + pycharm={"name": "#%%\n"}
from sklearn.metrics import accuracy_score


def evaluate_models(
    training_features, training_labels, test_features, test_labels, classifiers
):
    prediction_scores = {}

    for index, clf in enumerate(classifiers):
        clf.fit(training_features, training_labels)
        prediction = clf.predict(test_features)
        prediction_scores[clf.__class__.__name__] = accuracy_score(
            test_labels, prediction
        )
    return prediction_scores


class ExtraTreesClassifier_Finetuned(ExtraTreesClassifier):
    def __init__(self):
        ExtraTreesClassifier.__init__(
            self, n_estimators=102, max_depth=18, random_state=42
        )


classifiers = [
    ExtraTreesClassifier(),
    ExtraTreesClassifier_Finetuned(),
    LogisticRegression(),
    MLPClassifier(),
    SVC(),
    RandomForestClassifier(),
]

prediction_scores = evaluate_models(
    train_reduced_df, training_labels, test_reduced_df, test_labels, classifiers
)
prediction_scores

# + [markdown] pycharm={"name": "#%% md\n"}
# Plot the performance (accuracy) of the classifiers on the train vs test dataset.

# + pycharm={"name": "#%%\n"}

classifiers = [
    "ExtraTreesClassifier",
    "ExtraTreesClassifier_Finetuned",
    "LogisticRegression",
    "MLPClassifier",
    "SVC",
    "RandomForestClassifier",
]

N = len(classifiers)

train_results_df = pd.concat(
    [
        results,
        pd.DataFrame(
            [
                {
                    "classifier": "ExtraTreesClassifier_Finetuned",
                    "test_accuracy": searchCV.best_score_,
                }
            ]
        ),
    ]
)

train_results_df = train_results_df.set_index("classifier")

# Specify the values of blue bars (height)
blue_bar = train_results_df.loc[classifiers]["test_accuracy"].values
# Specify the values of orange bars (height)
orange_bar = [prediction_scores[c] for c in classifiers]

# Position of bars on x-axis
ind = np.arange(N)

# Figure size
plt.figure(figsize=(15, 8))

# Width of a bar
width = 0.3

# Plotting
plt.bar(ind, blue_bar, width, label="Train")
plt.bar(ind + width, orange_bar, width, label="Test")

plt.ylabel("Accuracy")
plt.title("Accuracy on train vs test sets")

plt.xticks(
    ind + width / 2,
    (
        "ExtraTreesClassifier",
        "ExtraTreesClassifier_Finetuned",
        "LogisticRegression",
        "MLPClassifier",
        "SVC",
        "RandomForestClassifier",
    ),
    rotation=90,
)


# Finding the best position for legends and putting it
plt.legend(loc="best")
plt.show()

# + [markdown] id="7l_q9R26jeQU" pycharm={"name": "#%% md\n"}
# ### Conclusion
#

# + [markdown] id="2FhonFTrm94x" pycharm={"name": "#%% md\n"}
# The prediction scores for the test set and the scores from the training were fairly close. The use of Cross Validation (and a high number of folds = 10) gave a good estimation of the performance on an unseen dataset.
#
# The dataset is very 'wide' with about 2 times the amount of rows/samples as there are features. We attempted two methods of feature/dimensionality reduction, RFE and PCA. RFE proved to work the best on the cross validated train data set.
#
# The best performing (in terms of accuracy) classifier on both train & test datasets was `ExtraTreesClassifier`, which seems to neither be over- or underfiting as the scores on both the training and test data are pretty close.
#
# The `LogisticRegression`, `MLPClassifier` and `SVC` classifiers all seems to be over- or underfitting slightly as their scores on the test set are a bit lower than on the training set.
#
# `RandomForestClassifier` is actually performing a bit better on the test set than on the training set.
#
# The tuned `ExtraTreesClassifier` does perform a bit better than the model with standard hyperparameters, it could perhaps be even better with some more tuning. On the other hand, more tuning could also risk overfitting it to the test set, which is why it's probably best to leave it as is.

# + [markdown] pycharm={"name": "#%% md\n"}
# # Task 3
