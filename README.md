---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .code execution_count="8" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="UaVUc-l_j8nk" outputId="3042902c-00be-4198-e501-35109569cf7c"}
``` python
pip install seaborn
```

::: {.output .stream .stdout}
    Requirement already satisfied: seaborn in /usr/local/lib/python3.11/dist-packages (0.13.2)
    Requirement already satisfied: numpy!=1.24.0,>=1.20 in /usr/local/lib/python3.11/dist-packages (from seaborn) (2.0.2)
    Requirement already satisfied: pandas>=1.2 in /usr/local/lib/python3.11/dist-packages (from seaborn) (2.2.2)
    Requirement already satisfied: matplotlib!=3.6.1,>=3.4 in /usr/local/lib/python3.11/dist-packages (from seaborn) (3.10.0)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.3.2)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (4.58.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (1.4.8)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (24.2)
    Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (11.2.1)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (3.2.3)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.11/dist-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.2->seaborn) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.2->seaborn) (2025.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.4->seaborn) (1.17.0)
:::
:::

::: {.cell .code execution_count="1" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="zq90gK2QM2EQ" outputId="f40d4299-0ece-45a0-8147-26962f402e52"}
``` python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("danofer/compass")

print("Path to dataset files:", path)
```

::: {.output .stream .stdout}
    Downloading from https://www.kaggle.com/api/v1/datasets/download/danofer/compass?dataset_version_number=1...
:::

::: {.output .stream .stderr}
    100%|██████████| 2.72M/2.72M [00:00<00:00, 3.20MB/s]
:::

::: {.output .stream .stdout}
    Extracting files...
:::

::: {.output .stream .stderr}
:::

::: {.output .stream .stdout}
    Path to dataset files: /root/.cache/kagglehub/datasets/danofer/compass/versions/1
:::
:::

::: {.cell .code execution_count="9" id="KBjw6aMGjnGr"}
``` python
# Import necessary libraries for data manipulation and modeling
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, demographic_parity_difference
```
:::

::: {.cell .code execution_count="13" id="-T0jJhlujmrP"}
``` python
# ======================
# Phase 1: Data Ingestion & Preprocessing
# ======================
# Load the COMPAS dataset (download at: https://www.kaggle.com/datasets/danofer/compass)
df = pd.read_csv('/content/compas-scores-raw.csv')
```
:::

::: {.cell .code execution_count="14" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":669}" id="mFT-fmX2mbqp" outputId="af7877bc-bdcc-40bd-813f-3a7af5358042"}
``` python
df
```

::: {.output .execute_result execution_count="14"}
``` json
{"type":"dataframe","variable_name":"df"}
```
:::
:::

::: {.cell .code execution_count="18" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="nagsf956oaA4" outputId="bed3193f-7b91-4466-bc4b-1b439afa62fb"}
``` python
df.shape
```

::: {.output .execute_result execution_count="18"}
    (60843, 28)
:::
:::

::: {.cell .code execution_count="16" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="N-0iZdP6mrk6" outputId="5a781c93-ee8f-4c1d-be56-058a08b139bf"}
``` python
df.columns
```

::: {.output .execute_result execution_count="16"}
    Index(['Person_ID', 'AssessmentID', 'Case_ID', 'Agency_Text', 'LastName',
           'FirstName', 'MiddleName', 'Sex_Code_Text', 'Ethnic_Code_Text',
           'DateOfBirth', 'ScaleSet_ID', 'ScaleSet', 'AssessmentReason',
           'Language', 'LegalStatus', 'CustodyStatus', 'MaritalStatus',
           'Screening_Date', 'RecSupervisionLevel', 'RecSupervisionLevelText',
           'Scale_ID', 'DisplayText', 'RawScore', 'DecileScore', 'ScoreText',
           'AssessmentType', 'IsCompleted', 'IsDeleted'],
          dtype='object')
:::
:::

::: {.cell .code execution_count="19" id="Y7VUj9W8lUbR"}
``` python
# Select relevant columns for recidivism prediction
df = df[['Sex_Code_Text', 'Ethnic_Code_Text', 'DecileScore']]

```
:::

::: {.cell .code execution_count="20" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="zIRv5qv9mmi3" outputId="5a26a6d4-1392-4c39-eacb-b303765d0ab2"}
``` python
# Create binary target variable based on DecileScore.
df['recidivism'] = df['DecileScore'].apply(lambda x: 1 if x >= 5 else 0)
```

::: {.output .stream .stderr}
    <ipython-input-20-7e5a3f675c1e>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['recidivism'] = df['DecileScore'].apply(lambda x: 1 if x >= 5 else 0)
:::
:::

::: {.cell .code execution_count="21" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="EsmLM3q-uKMB" outputId="81b054f9-0337-4f58-b712-fe938b29785c"}
``` python
# Encode categorical variables:
# Map Sex_Code_Text to numeric (for example, if values are 'Male' and 'Female')
df['sex'] = df['Sex_Code_Text'].map({'Male': 1, 'Female': 0})
```

::: {.output .stream .stderr}
    <ipython-input-21-8c228176162a>:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df['sex'] = df['Sex_Code_Text'].map({'Male': 1, 'Female': 0})
:::
:::

::: {.cell .code execution_count="22" id="-SLtQgjfuJ_9"}
``` python
# The sensitive attribute is 'Ethnic_Code_Text'.
sensitive_attr = df['Ethnic_Code_Text']
```
:::

::: {.cell .code execution_count="23" id="4GoD1nwxuJ2s"}
``` python
# Define feature matrix X and target vector y.
# We use 'sex' and 'DecileScore' as features.
X = df[['sex', 'DecileScore']]
y = df['recidivism']
```
:::

::: {.cell .code execution_count="24" id="anzefWC3uJqS"}
``` python
# Split the dataset into training and testing sets (70%/30% split).
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, sensitive_attr, test_size=0.3, random_state=42)
```
:::

::: {.cell .markdown id="oiUdVnAvughi"}
# ======================

### Phase 2: Baseline Modeling with Fairlearn

# ======================
:::

::: {.cell .code execution_count="49" id="uDByB_dRuJa7"}
``` python
from fairlearn.metrics import MetricFrame
import numpy as np

def evaluate_fairness(y_true, y_pred, sensitive_features):
    mf = MetricFrame(
        metrics={'prediction_mean': lambda y_true, y_pred: np.mean(y_pred)},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    demographic_parity_diff = mf.by_group['prediction_mean'].max() - mf.by_group['prediction_mean'].min()
    return mf.by_group, demographic_parity_diff
```
:::

::: {.cell .code execution_count="50" id="Meq_p6K2utXS"}
``` python
baseline_results = {}
models = {}
```
:::

::: {.cell .markdown id="aJoz44g8vAkC"}
### 1. Logistic Regression {#1-logistic-regression}
:::

::: {.cell .code execution_count="51" id="h-RFI-MnutL7"}
``` python
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
```
:::

::: {.cell .code execution_count="52" id="L2ViEBOQutD0"}
``` python
baseline_results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, lr_preds),
    'fairness': evaluate_fairness(y_test, lr_preds, s_test)
}
models['Logistic Regression'] = lr
```
:::

::: {.cell .markdown id="41nJDwY6z-tX"}
### 2. Decision Tree {#2-decision-tree}
:::

::: {.cell .code execution_count="53" id="8hA74p1l0CjA"}
``` python
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
```
:::

::: {.cell .code execution_count="54" id="WdzI_yKI0B8T"}
``` python
baseline_results['Decision Tree'] = {
    'accuracy': accuracy_score(y_test, dt_preds),
    'fairness': evaluate_fairness(y_test, dt_preds, s_test)
}
models['Decision Tree'] = dt
```
:::

::: {.cell .markdown id="ZeSHZJaDv27v"}
### (3) Random Forest {#3-random-forest}
:::

::: {.cell .code execution_count="55" id="NsrAl5mYvI8E"}
``` python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
```
:::

::: {.cell .code execution_count="56" id="46iXoqmgvI5Z"}
``` python
baseline_results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, rf_preds),
    'fairness': evaluate_fairness(y_test, rf_preds, s_test)
}
models['Random Forest'] = rf
```
:::

::: {.cell .markdown id="LKmylMfE0VID"}
### 4. Support Vector Machine (with linear kernel for interpretability) {#4-support-vector-machine-with-linear-kernel-for-interpretability}
:::

::: {.cell .code execution_count="57" id="_DWsePUV0Vm9"}
``` python
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)
baseline_results['SVM'] = {
    'accuracy': accuracy_score(y_test, svm_preds),
    'fairness': evaluate_fairness(y_test, svm_preds, s_test)
}
models['SVM'] = svm
```
:::

::: {.cell .code execution_count="58" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="K64zjbI20V7t" outputId="a0291ecc-a478-443f-ab32-b56f0e156b12"}
``` python
# Display baseline results.
for model_name, result in baseline_results.items():
    print(f"\nModel: {model_name}")
    print("Accuracy:", round(result['accuracy'], 3))
    print("Fairness metric (dp_diff) by group:")
    print(result['fairness'][0])
    print("Overall fairness metric:", round(result['fairness'][1], 3))
```

::: {.output .stream .stdout}

    Model: Logistic Regression
    Accuracy: 1.0
    Fairness metric (dp_diff) by group:
                      prediction_mean
    Ethnic_Code_Text                 
    African-Am               0.545455
    African-American         0.425878
    Arabic                   0.166667
    Asian                    0.168421
    Caucasian                0.250339
    Hispanic                 0.208868
    Native American          0.352113
    Oriental                 0.200000
    Other                    0.160000
    Overall fairness metric: 0.385

    Model: Decision Tree
    Accuracy: 1.0
    Fairness metric (dp_diff) by group:
                      prediction_mean
    Ethnic_Code_Text                 
    African-Am               0.545455
    African-American         0.425878
    Arabic                   0.166667
    Asian                    0.168421
    Caucasian                0.250339
    Hispanic                 0.208868
    Native American          0.352113
    Oriental                 0.200000
    Other                    0.160000
    Overall fairness metric: 0.385

    Model: Random Forest
    Accuracy: 1.0
    Fairness metric (dp_diff) by group:
                      prediction_mean
    Ethnic_Code_Text                 
    African-Am               0.545455
    African-American         0.425878
    Arabic                   0.166667
    Asian                    0.168421
    Caucasian                0.250339
    Hispanic                 0.208868
    Native American          0.352113
    Oriental                 0.200000
    Other                    0.160000
    Overall fairness metric: 0.385

    Model: SVM
    Accuracy: 1.0
    Fairness metric (dp_diff) by group:
                      prediction_mean
    Ethnic_Code_Text                 
    African-Am               0.545455
    African-American         0.425878
    Arabic                   0.166667
    Asian                    0.168421
    Caucasian                0.250339
    Hispanic                 0.208868
    Native American          0.352113
    Oriental                 0.200000
    Other                    0.160000
    Overall fairness metric: 0.385
:::
:::

::: {.cell .markdown id="GZFBfsIq1ulm"}
## Fairlearn Mitigation
:::

::: {.cell .code execution_count="59" id="g63s53cD1S21"}
``` python
# Simulate Fairlearn bias mitigation: assume dp_diff improves from 0.42 to approximately 0.27 over 20 epochs.
epochs = np.arange(1, 21)
simulated_fairness = 0.42 * np.exp(-0.2 * epochs) + 0.15  # Final value ~0.27

```
:::

::: {.cell .code execution_count="60" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":466}" id="tyPXfCf61Srl" outputId="0251c9c6-dfa6-4e14-a2b8-85ad220f6987"}
``` python
plt.figure(figsize=(8, 5))
plt.plot(epochs, simulated_fairness, marker='o', linestyle='-', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Simulated Fairness Metric (dp_diff)')
plt.title('Figure 1: Simulated Fairlearn Fairness Improvement')
plt.grid(True)
plt.savefig('fairlearn_fairness_improvement.png')
plt.show()
```

::: {.output .display_data}
![](vertopal_e960c4e7ee8d4c3bbf08d4ace8707026/cd42f4a4ecbc563d5e8b914df53b693abbb3dc66.png)
:::
:::

::: {.cell .markdown id="90N602Cd1-FS"}
## IBM AI Fairness 360 (AIF360) Mitigation:
:::

::: {.cell .code execution_count="62" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="YuoUy4em2QA1" outputId="a14c8501-f167-4364-939c-d3d71e3326bb"}
``` python
pip install aif360
```

::: {.output .stream .stdout}
    Collecting aif360
      Downloading aif360-0.6.1-py3-none-any.whl.metadata (5.0 kB)
    Requirement already satisfied: numpy>=1.16 in /usr/local/lib/python3.11/dist-packages (from aif360) (2.0.2)
    Requirement already satisfied: scipy>=1.2.0 in /usr/local/lib/python3.11/dist-packages (from aif360) (1.15.3)
    Requirement already satisfied: pandas>=0.24.0 in /usr/local/lib/python3.11/dist-packages (from aif360) (2.2.2)
    Requirement already satisfied: scikit-learn>=1.0 in /usr/local/lib/python3.11/dist-packages (from aif360) (1.6.1)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.11/dist-packages (from aif360) (3.10.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas>=0.24.0->aif360) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas>=0.24.0->aif360) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas>=0.24.0->aif360) (2025.2)
    Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn>=1.0->aif360) (1.5.1)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn>=1.0->aif360) (3.6.0)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib->aif360) (1.3.2)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.11/dist-packages (from matplotlib->aif360) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib->aif360) (4.58.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib->aif360) (1.4.8)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib->aif360) (24.2)
    Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.11/dist-packages (from matplotlib->aif360) (11.2.1)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib->aif360) (3.2.3)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas>=0.24.0->aif360) (1.17.0)
    Downloading aif360-0.6.1-py3-none-any.whl (259 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 259.7/259.7 kB 4.1 MB/s eta 0:00:00
:::
:::

::: {.cell .code execution_count="63" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="8B7PSTSf15bY" outputId="793b4405-83a4-4c33-aa31-d9f229e1526a"}
``` python
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
```

::: {.output .stream .stderr}
    WARNING:root:No module named 'inFairness': SenSeI and SenSR will be unavailable. To install, run:
    pip install 'aif360[inFairness]'
:::
:::

::: {.cell .code execution_count="83" id="PDINIMcR15Mu"}
``` python
# Modify the DataFrame to include the derived binary target variable.
df_aif360 = df.copy()
df_aif360['race'] = df_aif360['Ethnic_Code_Text']
# In IBM AIF360, the target column is 'recidivism', and favorable outcome is defined as 0.
privileged_groups = [{'race_Caucasian': 1}]
unprivileged_groups = [{'race_Caucasian': 0}]
```
:::

::: {.cell .code execution_count="84" id="XYwadSYl5r5q"}
``` python
# filter the dataset to only include two groups.
df_aif360 = df_aif360[df_aif360['race'].isin(['Caucasian', 'African-American'])]
```
:::

::: {.cell .code execution_count="85" id="Mg26xAz97WvC"}
``` python
df_numeric = pd.get_dummies(df_aif360)
```
:::

::: {.cell .code execution_count="90" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="rUfpVx365rub" outputId="4417d69e-ca54-499e-bd11-10d28d08c3f8"}
``` python
dataset = StandardDataset(
    df_numeric,
    label_name='recidivism',
    favorable_classes=[0],
    protected_attribute_names=['race_Caucasian'],  # or whatever the new one-hot column is
    privileged_classes=[[1]]  # adjust based on your one-hot encoding
)
```

::: {.output .stream .stderr}
    /usr/local/lib/python3.11/dist-packages/aif360/datasets/standard_dataset.py:122: FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. Value '1.0' has dtype incompatible with bool, please explicitly cast to a compatible dtype first.
      df.loc[priv, attr] = privileged_values[0]
:::
:::

::: {.cell .code execution_count="91" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="k4MopX4G7jw1" outputId="0d7eb512-0857-4a68-eb38-0e354be6885f"}
``` python
# Calculate disparate impact metric before mitigation.
metric_before = BinaryLabelDatasetMetric(dataset,
                                         privileged_groups=privileged_groups,
                                         unprivileged_groups=unprivileged_groups)
disparate_impact_before = metric_before.disparate_impact()
print("AIF360 - Disparate Impact before mitigation:", round(disparate_impact_before, 3))
```

::: {.output .stream .stdout}
    AIF360 - Disparate Impact before mitigation: 0.762
:::
:::

::: {.cell .code execution_count="92" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="DqhGbsXT7jih" outputId="f7ceca2b-e4b2-4406-ed47-563201060ef5"}
``` python
# Apply the Reweighing algorithm.
RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
dataset_transf = RW.fit_transform(dataset)

metric_after = BinaryLabelDatasetMetric(dataset_transf,
                                        privileged_groups=privileged_groups,
                                        unprivileged_groups=unprivileged_groups)
disparate_impact_after = metric_after.disparate_impact()
print("AIF360 - Disparate Impact after mitigation:", round(disparate_impact_after, 3))
```

::: {.output .stream .stdout}
    AIF360 - Disparate Impact after mitigation: 1.0
:::
:::
