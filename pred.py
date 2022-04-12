import argparse
import lightgbm as lgb
import numpy as np
import optuna
import os
import pandas as pd
import re
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split



parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=0.4, help="閾値を指定してください。")
parser.add_argument("--valid_size", type=float, default=0.2, help="検証データのサイを指定してください。")
parser.add_argument("--n_trials", type=int, default=1000, help="ハイパーパラメータ最適化の試行回数を指定してください。")

args = parser.parse_args()

os.makedirs('./Results/', exist_ok=True)


def get_ticket_num(s):
    num_list = re.findall(r"\d+", s)
    if len(num_list) == 0:
        return np.nan
    else:
        return int(num_list[-1])

def preprocessing(df):
    df.drop(columns=["Cabin", "Name"], inplace=True)
    df.dropna(subset={"Age", "Embarked"}, inplace=True)

    df['Num of Ticket'] = df['Ticket'].map(get_ticket_num)
    df.drop(columns={"Ticket"}, inplace=True)

    df = pd.get_dummies(df, drop_first=True)

    return df


def prob_to_labels(ls, thr=args.threshold):
    return np.where(ls>thr, 1, 0)

def objective(trial: optuna.trial):
    num_leaves = trial.suggest_int('num_leaves', 2, 30)
    n_estimators = trial.suggest_int('n_estimators', 1, 30)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1)
    reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-3, 1)

    lgb_params = {
        'num_leaves':num_leaves,
        'objective':'binary',
        'n_estimators':n_estimators,
        'learning_rate':learning_rate,
        'reg_alpha': reg_alpha,
    }

    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    lgb_clf.fit(X_train, labels_train)
    pred_train = lgb_clf.predict_proba(X_train)[:, 1]
    pred_valid = lgb_clf.predict_proba(X_valid)[:, 1]

    return roc_auc_score(labels_valid, pred_valid)


if __name__ == '__main__':
    train_df = pd.read_csv('./Data/train.csv')
    test_df = pd.read_csv('./Data/test.csv')


    train_df = preprocessing(train_df)
    test_df = preprocessing(test_df)

    X_train_base = train_df.drop(columns="Survived")
    labels_train_base = train_df["Survived"]

    X_train, X_valid, labels_train, labels_valid = train_test_split(X_train_base, labels_train_base, test_size=args.valid_size)
    id_train = X_train["PassengerId"]
    X_train = X_train.drop(columns="PassengerId")
    id_valid = X_valid["PassengerId"]
    X_valid = X_valid.drop(columns="PassengerId")

    id_test = test_df["PassengerId"]
    X_test = test_df.drop(columns="PassengerId")


    sampler = optuna.samplers.CmaEsSampler()
    #sampler = optuna.samplers.RandomSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=args.n_trials)


    lgb_params = {
            'num_leaves':study.best_params["num_leaves"],
            'objective':'binary',
            'n_estimators':study.best_params["n_estimators"],
            'learning_rate':study.best_params["learning_rate"],
            'reg_alpha':study.best_params["reg_alpha"],
        }

    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    lgb_clf.fit(X_train, labels_train)

    importances_df = pd.DataFrame({"features":X_train.columns, "importances": lgb_clf.feature_importances_})
    importances_df.sort_values("importances", ascending=False)
    importances_df.to_csv("./Results/feature_importances_df.csv", index=False)

    pred_train = lgb_clf.predict_proba(X_train)[:, 1]
    pred_labels_train = prob_to_labels(pred_train)

    pred_valid = lgb_clf.predict_proba(X_valid)[:, 1]
    pred_labels_valid = prob_to_labels(pred_valid)
    pred_test = lgb_clf.predict_proba(X_test)[:, 1]
    pred_labels_test = prob_to_labels(pred_test)

    results_df = pd.DataFrame({"PassengerId":id_test, "Survived": pred_labels_test})

    results_df.to_csv("./Results/result.csv", index=False)

    print('Train    ROC   : {}'.format(roc_auc_score(labels_train, pred_train)))
    print('Valid    ROC   : {}'.format(roc_auc_score(labels_valid, pred_valid)))
    print('Train Accuracy : {}'.format(accuracy_score(labels_train, pred_labels_train)))
    print('Valid Accuracy : {}'.format(accuracy_score(labels_valid, pred_labels_valid)))
