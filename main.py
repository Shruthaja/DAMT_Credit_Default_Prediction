import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import optuna

from sklearn.model_selection import train_test_split
import sklearn.metrics

from xgboost import XGBClassifier

import cupy, cudf  # GPU libraries
import matplotlib.pyplot as plt, gc, os

import gc
import warnings

warnings.filterwarnings('ignore')


def read_train_file(path='', usecols=None):
    # LOAD DATAFRAME
    if usecols is not None:
        df = cudf.read_parquet(path, columns=usecols)
    else:
        df = cudf.read_parquet(path)
    # REDUCE DTYPE FOR CUSTOMER AND DATE
    df['customer_ID'] = df['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    df.S_2 = cudf.to_datetime(df.S_2)
    df = df.fillna(0)
    print('shape of data:', df.shape)

    return df


print('Reading train data...')
TRAIN_PATH = '../input/amex-data-integer-dtypes-parquet-format/train.parquet'
train = read_train_file(path=TRAIN_PATH)


def process_and_feature_engineer(df):
    all_cols = [c for c in list(df.columns) if c not in ['customer_ID', 'S_2']]
    cat_features = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]
    num_features = [col for col in all_cols if col not in cat_features]

    test_num_agg = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last'])
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]

    test_cat_agg = df.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
    test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]

    df = cudf.concat([test_num_agg, test_cat_agg], axis=1)
    del test_num_agg, test_cat_agg
    print('shape after engineering', df.shape)

    return df


train = process_and_feature_engineer(train)
targets = cudf.read_csv('../input/amex-default-prediction/train_labels.csv')
targets['customer_ID'] = targets['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
targets = targets.set_index('customer_ID')
train = train.merge(targets, left_index=True, right_index=True, how='left')
train.target = train.target.astype('int8')
del targets

train = train.sort_index().reset_index()

# FEATURES
FEATURES = train.columns[1:-1]
print(f'There are {len(FEATURES)} features!')

train_pd = train.to_pandas()
del train
_ = gc.collect()

train_df, test_df = train_test_split(train_pd, test_size=0.25, stratify=train_pd['target'])
del train_pd
_ = gc.collect()
len(train_df), len(test_df)

X_train = train_df.drop(['customer_ID', 'target'], axis=1)
X_test = test_df.drop(['customer_ID', 'target'], axis=1)

X_train.head()

y_train = train_df['target']
y_test = test_df['target']

y_train.head()

del train_df, test_df
_ = gc.collect()


def objective(trial):
    param = {
        'booster': 'gbtree',
        'tree_method': 'gpu_hist',
        "objective": "binary:logistic",
        'lambda': trial.suggest_loguniform(
            'lambda', 1e-3, 10.0
        ),
        'alpha': trial.suggest_loguniform(
            'alpha', 1e-3, 10.0
        ),
        'colsample_bytree': trial.suggest_float(
            'colsample_bytree', 0.5, 1, step=0.1
        ),
        'subsample': trial.suggest_float(
            'subsample', 0.5, 1, step=0.1
        ),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.001, 0.05, step=0.001
        ),
        'n_estimators': trial.suggest_int(
            "n_estimators", 80, 1000, 10
        ),
        'max_depth': trial.suggest_int(
            'max_depth', 2, 10, 1
        ),
        'random_state': 99,
        'min_child_weight': trial.suggest_int(
            'min_child_weight', 1, 256, 1
        ),
    }

    model = XGBClassifier(**param, enable_categorical=True)

    model.fit(X_train, y_train)

    preds = pd.DataFrame(model.predict(X_test))

    accuracy = sklearn.metrics.accuracy_score(pd.DataFrame(y_test.reset_index()['target']), preds)

    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)
best_params = study.best_trial.params
best_params['tree_method'] = 'gpu_hist'
best_params['booster'] = 'gbtree'

final_model = XGBClassifier(**best_params, enable_categorical=True)

final_model.fit(X_train, y_train)
result = final_model.predict_proba(X_test)[:, 1]

result = np.array([1 if i > 0.5 else 0 for i in result])

len(y_test), len(result)

cm = np.zeros((2, 2))

count = 0
for tval, pval in zip(y_test, result):
    count += 1
    cm[tval][pval] += 1

count, cm

import seaborn as sns

ax = sns.heatmap(cm)

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])

plt.show()

final = pd.DataFrame(test['prediction'].to_pandas())

final.to_csv("submission.csv", index=True)
