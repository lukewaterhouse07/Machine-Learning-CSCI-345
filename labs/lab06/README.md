# Lab | Bootstrapping, Bagging & Feature Importance

## Overview

A single machine learning model can be sensitive to the particular training data it receives. Ensemble methods address this problem by combining multiple models to produce predictions that are often more stable and reliable.

In this lab, you'll explore **bootstrapping and bagging** using the Breast Cancer Wisconsin (Diagnostic) dataset. You will begin with a single decision tree and then investigate what happens when we train multiple models on different bootstrap samples.

You will then extend this idea to **Random Forests**, which combine bootstrap sampling with random feature selection.

Finally, you'll investigate **feature importance** using two approaches:

- **Mean Decrease in Impurity (MDI)**
- **Permutation importance**

Throughout the lab, you'll compare these approaches and think critically about what feature importance actually tells us.

> **Important:** This lab uses a medical dataset for machine-learning practice. The models you build are for educational purposes and should not be interpreted as clinical diagnostic systems.

---

## Learning Goals

By the end of this lab, you should be able to:

- Explain what bootstrapping means.
- Create bootstrap samples using sampling with replacement.
- Explain the difference between bootstrapping and bagging.
- Build a `BaggingClassifier` using scikit-learn.
- Compare a single decision tree with a bagged ensemble.
- Explain how bagging can reduce variance.
- Build and evaluate a Random Forest classifier.
- Explain how Random Forest extends the bagging idea.
- Calculate and visualize MDI feature importance.
- Calculate and visualize permutation importance.
- Compare MDI and permutation importance.
- Explain why feature importance does not imply causation.
- Recognize how correlated features can complicate feature importance.

---

## Setup and Context

You'll work inside a Jupyter Notebook for this lab.

All analysis, code, visualizations, and written interpretations should be contained in a single notebook. Your notebook should show both **what you did** and **what you learned from the results**.

The dataset you'll use is the **Breast Cancer Wisconsin (Diagnostic)** dataset available through scikit-learn.

The dataset contains measurements computed from digitized images of breast mass samples. The target contains two classes:

- `0` → malignant
- `1` → benign

The dataset contains **30 numerical features** describing characteristics of the cell nuclei.

For this lab, the goal is to predict the target class using several individual and ensemble models.

---

## Requirements

### Python environment

Install the required packages if necessary:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Getting Started

Create a new Jupyter Notebook called:

```
lab-ensemble-methods.ipynb
```

Start with the following imports:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    VotingClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

sns.set_style("whitegrid")
```

Work through the tasks in order.

Use **Markdown cells** between code cells to explain your observations and reasoning.

---

# Task 1: Load and Explore the Dataset

Before building models, we need to understand the data.

### 1\. Load the dataset

Use scikit-learn to load the Breast Cancer Wisconsin dataset:

```
data = load_breast_cancer(as_frame=True)

X = data.data
y = data.target
```

Inspect the first few rows:

```
X.head()
```

Inspect the target:

```
y.head()
```

### 2\. Explore the dataset

Use appropriate pandas commands to determine:

- Number of observations
- Number of features
- Feature names
- Data types
- Whether missing values exist

### Questions

Answer the following in a Markdown cell:

1. How many observations are in the dataset?
2. How many predictor variables are available?
3. Are the features numerical or categorical?
4. Are there any missing values?
5. How many observations belong to each target class?
6. Is the target balanced or imbalanced?

---

# Task 2: Understand the Target Variable

The dataset provides numerical target values.

Use:

```
print(data.target_names)
```

Then examine the target distribution:

```
y.value_counts()
```

Create a table showing the class names and their counts.

### Optional Visualization

Create a bar chart showing the number of observations in each class.

### Questions

1. What does class `0` represent?
2. What does class `1` represent?
3. Which class has more observations?
4. Why should we consider class balance when evaluating a classification model?

---

# Task 3: Split the Data

Split the data into training and test sets.

Use an 80/20 split:

```
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

### Questions

Report:

- Number of training observations
- Number of test observations
- Number of features

Explain why we use:

```
stratify=y
```

when creating the train/test split.

---

# Task 4: Build a Single Decision Tree

Before using bagging, we need a baseline model.

Train a single decision tree:

```
tree = DecisionTreeClassifier(
    random_state=42
)

tree.fit(X_train, y_train)
```

Generate predictions:

```
y_pred_tree = tree.predict(X_test)
```

Calculate accuracy:

```
tree_accuracy = accuracy_score(
    y_test,
    y_pred_tree
)

print("Accuracy:", tree_accuracy)
```

Calculate F1:

```
tree_f1 = f1_score(
    y_test,
    y_pred_tree
)

print("F1:", tree_f1)
```

Also display the classification report:

```
print(
    classification_report(
        y_test,
        y_pred_tree,
        target_names=data.target_names
    )
)
```

### Questions

1. What accuracy did the decision tree achieve?
2. What F1 score did it achieve?
3. Why is it useful to establish a baseline before building an ensemble?
4. What does it mean for a model to have high variance?

---

# Task 5: Understand Bootstrapping

Before using a bagging model, let's create bootstrap samples manually.

Consider this simple dataset:

```
observations = np.array([
    "A", "B", "C", "D", "E"
])
```

Create a bootstrap sample:

```
bootstrap_sample = np.random.choice(
    observations,
    size=len(observations),
    replace=True
)

print(bootstrap_sample)
```

Run the code several times.

### Questions

Observe the results and answer:

1. Does every observation appear in every bootstrap sample?
2. Can an observation appear more than once?
3. Does each bootstrap sample contain five observations?
4. What does `replace=True` mean?
5. Why are the bootstrap samples different?

---

# Task 6: Create Multiple Bootstrap Samples

Create five bootstrap samples:

```
for i in range(5):

    sample = np.random.choice(
        observations,
        size=len(observations),
        replace=True
    )

    print(
        f"Sample {i + 1}: {sample}"
    )
```

### Questions

1. How are the samples different?
2. Do all samples contain the same observations?
3. Why might these differences be useful when training multiple models?

### Written Explanation

In a Markdown cell, explain:

> **How does sampling with replacement allow us to create different training datasets from the same original dataset?**

---

# Task 7: Bootstrap the Training Data

Now apply bootstrapping to the actual training data.

First combine the predictors and target:

```
train_data = X_train.copy()

train_data["target"] = y_train.values
```

Create a bootstrap sample:

```
bootstrap_sample = train_data.sample(
    n=len(train_data),
    replace=True,
    random_state=42
)
```

Inspect:

```
bootstrap_sample.head()
```

### Investigate the Sample

Calculate the number of unique observations:

```
bootstrap_sample.index.nunique()
```

Compare this with:

```
len(bootstrap_sample)
```

### Questions

1. Does the bootstrap sample have the same number of rows as the original training set?
2. Does it have the same number of unique observations?
3. Why are these numbers potentially different?
4. What does this tell you about sampling with replacement?

---

# Task 8: Build a Bagging Classifier

Now let scikit-learn create bootstrap samples and train multiple models for us.

Create a bagging classifier using decision trees:

```
bagging_tree = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        random_state=42
    ),
    n_estimators=100,
    random_state=42
)
```

Train the model:

```
bagging_tree.fit(
    X_train,
    y_train
)
```

Generate predictions:

```
y_pred_bagging = bagging_tree.predict(
    X_test
)
```

Calculate accuracy and F1:

```
bagging_accuracy = accuracy_score(
    y_test,
    y_pred_bagging
)

bagging_f1 = f1_score(
    y_test,
    y_pred_bagging
)

print("Accuracy:", bagging_accuracy)
print("F1:", bagging_f1)
```

---

# Task 9: Compare One Tree vs. Bagging

Create a comparison table:

| Model | Accuracy | F1 |
| --- | --- | --- |
| Decision Tree | ... | ... |
| Bagging | ... | ... |

### Questions

1. Which model performed better?
2. Did bagging improve accuracy?
3. Did bagging improve F1?
4. Was the improvement large or small?
5. Why might combining multiple decision trees make predictions more stable?

### Key Concept

Remember:

> **Bagging primarily helps reduce variance.**

It does not guarantee that every dataset will show a dramatic performance improvement.

---

# Task 10: Investigate the Number of Estimators

Bagging combines predictions from multiple models.

Let's investigate how the number of models affects performance.

Train bagging classifiers using:

```
1
5
10
25
50
100
```

estimators.

For each value:

1. Create a `BaggingClassifier`.
2. Train it.
3. Generate predictions.
4. Calculate accuracy.
5. Calculate F1.
6. Store the results.

Create a DataFrame similar to:

| Number of Estimators | Accuracy | F1 |
| --- | --- | --- |
| 1 | ... | ... |
| 5 | ... | ... |
| 10 | ... | ... |
| 25 | ... | ... |
| 50 | ... | ... |
| 100 | ... | ... |

### Visualization

Create a line plot showing accuracy against the number of estimators.

Then create another line plot showing F1 against the number of estimators.

### Questions

1. What happens as the number of estimators increases?
2. Does performance improve indefinitely?
3. At what point does performance appear to become relatively stable?
4. What is the trade-off between using more estimators and computational cost?

---

# Task 11: Bagging with Different Base Models

Bagging can be used with different base estimators.

Your lecture demonstrated that `BaggingClassifier` can use models such as:

- Decision Trees
- Logistic Regression
- K-Nearest Neighbors

Build the following models.

### Bagged Decision Tree

```
bag_tree = BaggingClassifier(
    estimator=DecisionTreeClassifier(
        random_state=42
    ),
    n_estimators=100,
    random_state=42
)
```

### Bagged Logistic Regression

```
bag_logistic = BaggingClassifier(
    estimator=LogisticRegression(
        max_iter=1000
    ),
    n_estimators=100,
    random_state=42
)
```

### Bagged KNN

```
bag_knn = BaggingClassifier(
    estimator=KNeighborsClassifier(),
    n_estimators=100,
    random_state=42
)
```

Train and evaluate all three.

Create a table:

| Model | Accuracy | F1 |
| --- | --- | --- |
| Bagged Decision Tree | ... | ... |
| Bagged Logistic Regression | ... | ... |
| Bagged KNN | ... | ... |

### Questions

1. Which base model performed best with bagging?
2. Did bagging appear equally useful for all three models?
3. Which model benefited most from the ensemble?
4. Why might bagging be particularly useful for models with high variance?

---

# Task 12: Build a Random Forest

Random Forest extends the bagging idea.

Create a Random Forest classifier:

```
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

Train it:

```
rf.fit(
    X_train,
    y_train
)
```

Generate predictions:

```
y_pred_rf = rf.predict(
    X_test
)
```

Calculate accuracy and F1:

```
rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
)

rf_f1 = f1_score(
    y_test,
    y_pred_rf
)

print("Accuracy:", rf_accuracy)
print("F1:", rf_f1)
```

---

# Task 13: Bagging vs. Random Forest

Create a comparison table:

| Model | Accuracy | F1 |
| --- | --- | --- |
| Decision Tree | ... | ... |
| Bagged Decision Trees | ... | ... |
| Random Forest | ... | ... |

### Questions

Explain:

1. What do Bagging and Random Forest have in common?
2. What additional randomness does Random Forest introduce?
3. How does random feature selection make the trees less correlated?
4. Which model performed best?
5. Why might reducing correlation between trees help the ensemble?

### Key Concept

> **Random Forest = Bagged decision trees + random feature selection**

---

# Task 14: MDI Feature Importance

A Random Forest contains many decision trees.

Each tree makes predictions using splits based on features.

Scikit-learn provides feature importance through:

```
rf.feature_importances_
```

Create a pandas Series:

```
importance_mdi = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
).sort_values(
    ascending=False
)

print(importance_mdi)
```

Check the total:

```
importance_mdi.sum()
```

The values should sum to approximately `1`.

### Questions

1. Which feature has the highest MDI importance?
2. Which three features are most important?
3. Which feature has the lowest importance?
4. What does a high MDI importance indicate?
5. What role do tree splits play in determining MDI importance?

---

# Task 15: Visualize MDI Feature Importance

Create a horizontal bar chart:

```
importance_mdi.sort_values().plot(
    kind="barh",
    figsize=(9, 8)
)

plt.xlabel(
    "Feature Importance"
)

plt.title(
    "Random Forest Feature Importance — MDI"
)

plt.show()
```

### Questions

Based on your visualization:

1. Which features does the model rely on most?
2. Are a few features responsible for a large proportion of the importance?
3. Are there features with very small importance values?
4. Does a low importance score necessarily mean the feature is useless?

---

# Task 16: Interpret Feature Importance Carefully

Suppose your model reports:

```
worst radius → importance = 0.30
```

A student says:

> "Worst radius is the most important feature, so worst radius causes a tumor to be malignant."

### Question

Is this conclusion justified?

Explain why or why not.

Your answer should distinguish between:

- Predictive usefulness
- Association
- Causation

### Key Principle

> **Feature importance tells us what the model found useful for prediction—not necessarily what causes the outcome.**

---

# Task 17: Permutation Importance

MDI is not the only way to measure feature importance.

Permutation importance asks:

> **What happens to model performance when we randomly shuffle a feature?**

Calculate permutation importance for the Random Forest:

```
result = permutation_importance(
    rf,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)
```

Create a Series:

```
importance_permutation = pd.Series(
    result.importances_mean,
    index=X_test.columns
).sort_values(
    ascending=False
)

print(importance_permutation)
```

### Questions

1. Which feature has the highest permutation importance?
2. Which three features are most important?
3. Are the results identical to the MDI results?
4. Which features changed their ranking?
5. Are any features close to zero?

---

# Task 18: Visualize Permutation Importance

Create a horizontal bar chart:

```
importance_permutation.sort_values().plot(
    kind="barh",
    figsize=(9, 8)
)

plt.xlabel(
    "Mean Decrease in Model Performance"
)

plt.title(
    "Random Forest Feature Importance — Permutation"
)

plt.show()
```

### Questions

1. Which features have the largest performance impact when shuffled?
2. Which features appear least important?
3. How does this visualization compare with the MDI visualization?

---

# Task 19: Compare MDI and Permutation Importance

Create a DataFrame comparing both methods:

```
importance_comparison = pd.DataFrame({
    "MDI": importance_mdi,
    "Permutation": importance_permutation
})

importance_comparison.sort_values(
    "MDI",
    ascending=False
)
```

### Complete the following table

| Question | Answer |
| --- | --- |
| MDI is based on... | ... |
| Permutation importance is based on... | ... |
| MDI looks at... | ... |
| Permutation importance looks at... | ... |
| One limitation of MDI is... | ... |
| One advantage of permutation importance is... | ... |

### Analysis

Answer:

1. Which features are ranked highly by both methods?
2. Which features have different rankings?
3. Why might the two methods disagree?
4. Which method directly measures the effect of a feature on model performance?
5. Why might permutation importance take longer to calculate than MDI?

---

# Task 20: Investigate Correlated Features

The lecture discussed an important limitation of feature importance: **correlated features**.

Create a correlation matrix:

```
plt.figure(
    figsize=(12, 10)
)

sns.heatmap(
    X.corr(),
    cmap="coolwarm",
    center=0
)

plt.title(
    "Feature Correlations"
)

plt.show()
```

Identify at least one pair of strongly correlated features.

### Questions

1. Which pair of features did you identify?
2. Why can correlated features make feature importance difficult to interpret?
3. If two features contain similar information, how might their importance be distributed?
4. Could a feature have low importance even though it contains useful information?
5. What does this tell you about interpreting feature importance?

---

# Task 21: Build a Voting Classifier

Your lecture also introduced `VotingClassifier`.

Build a voting classifier using:

- Decision Tree
- Logistic Regression
- KNN

```
voting_model = VotingClassifier(
    estimators=[
        (
            "tree",
            DecisionTreeClassifier(
                random_state=42
            )
        ),
        (
            "logistic",
            LogisticRegression(
                max_iter=1000
            )
        ),
        (
            "knn",
            KNeighborsClassifier()
        )
    ],
    voting="hard"
)
```

Train the model:

```
voting_model.fit(
    X_train,
    y_train
)
```

Generate predictions:

```
y_pred_voting = voting_model.predict(
    X_test
)
```

Calculate accuracy and F1.

### Questions

1. How does hard voting make a prediction?
2. How is voting different from bagging?
3. Does the voting classifier outperform the individual models?
4. Why might combining different types of models be useful?

---

# Task 22: Final Model Comparison

Create one final DataFrame containing the major models built throughout the lab.

Include:

- Decision Tree
- Bagged Decision Tree
- Bagged Logistic Regression
- Bagged KNN
- Random Forest
- Voting Classifier

The table should contain:

- Accuracy
- F1 score

For example:

| Model | Accuracy | F1 |
| --- | --- | --- |
| Decision Tree | ... | ... |
| Bagged Decision Tree | ... | ... |
| Bagged Logistic Regression | ... | ... |
| Bagged KNN | ... | ... |
| Random Forest | ... | ... |
| Voting Classifier | ... | ... |

Sort the table by F1 score from highest to lowest.

---

# Task 23: Final Analysis

Complete the following questions in a detailed Markdown cell.

## Bootstrapping

1. What is bootstrapping?
2. Why are observations repeated in a bootstrap sample?
3. Why might some observations be missing from a bootstrap sample?
4. Why does a bootstrap sample have the same number of observations as the original dataset?

## Bagging

5. What does bagging stand for?
6. How does bagging use bootstrap samples?
7. How does bagging combine model predictions?
8. Why can bagging reduce variance?
9. Why is bagging particularly useful for high-variance models?

## Random Forest

10. How does Random Forest extend bagging?
11. Why does Random Forest randomly select features?
12. Why is reducing the correlation between trees useful?
13. What is the difference between Bagging and Random Forest?

## Feature Importance

14. What does MDI measure?
15. What does permutation importance measure?
16. Why might MDI and permutation importance produce different rankings?
17. Why can correlated features complicate feature importance?
18. Why does feature importance not imply causation?

## Model Selection

19. Which model performed best in your experiment?
20. How did the best ensemble compare with the single decision tree?
21. Did increasing the number of estimators always improve performance?
22. Which base model benefited most from bagging?
23. What are the advantages of using an ensemble?
24. What are the disadvantages?
25. If you were selecting a model for a real application, would you choose the model with the highest accuracy automatically? Explain.

---

# Submission

## What to Submit

Submit:

```
lab-ensemble-methods.ipynb
```

Your notebook should contain:

- All code
- All outputs
- All visualizations
- Markdown explanations
- Answers to the questions
- Final model comparison

---

# Definition of Done

Before submitting, verify that:

- [ ]The Breast Cancer Wisconsin dataset is loaded using scikit-learn.
- [ ]The dataset structure and target distribution are explored.
- [ ]The training and test sets are created correctly.
- [ ]A single Decision Tree baseline is established.
- [ ]Bootstrap samples are created manually.
- [ ]Sampling with replacement is demonstrated and explained.
- [ ]A bootstrap sample is created from the training data.
- [ ]A `BaggingClassifier` is trained.
- [ ]Bagging is compared with a single Decision Tree.
- [ ]Different numbers of estimators are investigated.
- [ ]Bagging is tested with Decision Tree, Logistic Regression, and KNN base models.
- [ ]A Random Forest is trained and evaluated.
- [ ]Bagging and Random Forest are compared.
- [ ]MDI feature importance is calculated.
- [ ]MDI feature importance is visualized.
- [ ]The meaning and limitations of MDI are discussed.
- [ ]Permutation importance is calculated.
- [ ]Permutation importance is visualized.
- [ ]MDI and permutation importance are compared.
- [ ]Feature correlations are investigated.
- [ ]The effect of correlated features on importance is discussed.
- [ ]A `VotingClassifier` is trained and evaluated.
- [ ]A final comparison table includes the major models.
- [ ]The final analysis questions are answered.
- [ ]The notebook runs from beginning to end without errors.

---

# Key Concepts to Remember

### Bootstrapping

> **Create new datasets by sampling observations with replacement.**

### Bagging

> **Train multiple models on bootstrap samples and aggregate their predictions.**

### Variance Reduction

> **Combining multiple models can make predictions more stable and reduce variance.**

### Random Forest

> **Bagged decision trees + random feature selection.**

### MDI

> **Measures how much features contribute to impurity reduction in decision trees.**

### Permutation Importance

> **Measures how much model performance decreases when a feature is shuffled.**

### Feature Importance

> **Feature importance describes predictive usefulness; it does not establish causation.**
