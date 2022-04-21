import argparse
import lightgbm as lgb
import logging
import matplotlib.pyplot as plt
import numpy as np
import optuna
import os
import pandas as pd
import re
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
import sys

from Module import *

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size4age", type=int, default=64, help="batch size (integer) for predicting \"Age\"")
parser.add_argument("--learning_rate4age", type=float, default=5e-3, help="learning rate for predicting \"Age\"")
parser.add_argument("--train_size_rate4age", type=float, default=0.7, help="rate of train size for predicting \"Age\" (0<.<1)")
parser.add_argument("--threshold", type=float, default=0.5, help="threshold for changing  probability tp \"Survived\" labels")
parser.add_argument("--n_trials", type=int, default=1000, help="number of hyperparameter optimization trials (integer)")
parser.add_argument("--n_split", type=int, default=5, help="number of split for cross validation")

#GlobalSetting
args = parser.parse_args()
kf = KFold(n_splits=args.n_split)
os.makedirs('./Results/', exist_ok=True)
results_path = "./Results/result.csv"

def get_ticket_num(s):
    num_list = re.findall(r"\d+", s)
    if len(num_list) == 0:
        return np.nan
    else:
        return int(num_list[-1])

def preprocessing(batch_size, leraning_rate, train_size_rate):
    all_data = pd.concat([train_data, test_data])

    #delete "Cabin" and "Name"
    all_data.drop(columns=["Name", "Cabin"], inplace=True)

    #pick up ticket number from "Ticket"
    all_data["Ticket"] = all_data["Ticket"].map(get_ticket_num)
    #standardlize ticket number
    all_data["Ticket"] = standardlization(all_data["Ticket"])
    #fill missing values of ticket number with 0
    all_data["Ticket"].fillna(0, inplace=True)

    #fill missing values of "Embarked" with mode
    all_data["Embarked"].fillna(all_data["Embarked"].value_counts().sort_values().index[-1], inplace=True)

    #standardlize "Fare"
    all_data["Fare"] = standardlization(all_data["Fare"])
    #fill missing values of "Fare" with 0
    all_data["Fare"].fillna(0, inplace=True)

    #predict "Age"
    logger.info("Start to predict [Age]")
    all_df = pred_age(all_data, batch_size, leraning_rate, train_size_rate)
    logger.info("Finish predicting [Age]")
    return all_df

def objective(trial: optuna.trial):
    num_leaves = trial.suggest_int('num_leaves', 2, 30)
    n_estimators = trial.suggest_int('n_estimators', 1, 30)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
    lambda_l1 = trial.suggest_loguniform('lambda_l1', 1e-3, 1e-1)
    lambda_l2 = trial.suggest_loguniform('lambda_l2', 1e-3, 1e-1)

    lgb_params = {
        'num_leaves':num_leaves,
        'objective':'binary',
        'n_estimators':n_estimators,
        'learning_rate':learning_rate,
        "lambda_l1":lambda_l1,
        "lambda_l2":lambda_l2,
        "verbose":-1,
    }

    X_base = train_df.drop(columns=["PassengerId", "Survived"]).values
    labels_base = train_df["Survived"].values
    cv = 0
    for _fold, (train_index, valid_index) in enumerate(kf.split(X_base)):
        labels_train = labels_base[train_index]
        X_train = X_base[train_index,:]
        labels_valid = labels_base[valid_index]
        X_valid = X_base[valid_index,:]

        lgb_clf = lgb.LGBMClassifier(**lgb_params)
        lgb_clf.fit(X_train, labels_train)
        pred_train = lgb_clf.predict_proba(X_train)[:, 1]
        pred_valid = lgb_clf.predict_proba(X_valid)[:, 1]

        cv += roc_auc_score(labels_valid, pred_valid) / kf.n_splits
    return cv

def prob_to_labels(ls, thr):
    return np.where(ls>thr, 1, 0)

def kill_logger(logger):
    name = logger.name
    del logging.Logger.manager.loggerDict[name]
    return

def kill_handler(logger):
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()
    return

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    train_data = pd.read_csv("./Data/train.csv")
    test_data = pd.read_csv("./Data/test.csv")

    logger.info("Start to preprocess")
    all_df = preprocessing(args.batch_size4age, args.learning_rate4age, args.train_size_rate4age)
    logger.info("finish preprocessing")

    train_df = all_df[~(all_df["Survived"].isna())]
    test_df = all_df[all_df["Survived"].isna()]

    sampler = optuna.samplers.CmaEsSampler()
    study = optuna.create_study(sampler=sampler, direction="maximize")
    logger.info("Start to optimize hyperparameter [Survived]")
    study.optimize(objective, n_trials=args.n_trials)
    logger.info("end optimizing hyperparameter [Survived]")

    lgb_params = {
        'num_leaves':study.best_params["num_leaves"],
        'objective':'binary',
        'n_estimators':study.best_params["n_estimators"],
        'learning_rate':study.best_params["learning_rate"],
        "lambda_l1":study.best_params["lambda_l1"],
        "lambda_l2":study.best_params["lambda_l2"],
        "verbose":-1,
    }

    logger.info("best score : {}".format(study.best_value))
    logger.info("best parameters : {}".format(study.best_params))

    X_base = train_df.drop(columns=["PassengerId", "Survived"]).values
    labels_base = train_df["Survived"].values

    lgb_clf_list = []
    for fold, (train_index, valid_index) in enumerate(kf.split(X_base)):
        logger.info("predict by clffifier No.{0}/{1} [Survived]".format(fold+1, args.n_split))
        part_train_loss_list = []

        labels_train = labels_base[train_index]
        X_train = X_base[train_index,:]
        labels_valid = labels_base[valid_index]
        X_valid = X_base[valid_index,:]

        lgb_clf = lgb.LGBMClassifier(**lgb_params)
        lgb_clf.fit(X_train, labels_train)

        lgb_clf_list.append(lgb_clf)


    id_test = test_df["PassengerId"]
    X_test = test_df.drop(columns=["PassengerId", "Survived"])

    logger.info("ensemble methods [Survived]")

    pred_dict = {}
    for num_clf, lgb_clf in enumerate(lgb_clf_list):
        pred = lgb_clf.predict_proba(X_test) [:, 1]
        pred_dict["clf{}".format(num_clf)] = pred

    pred_dict["PassengerId"] = id_test

    pred_df = pd.DataFrame(pred_dict)
    pred_df["clf_mean"] = pred_df.drop(columns="PassengerId").mean(axis=1)
    pred_labels_test = prob_to_labels(pred_df["clf_mean"], args.threshold)

    results_df = pd.DataFrame({"PassengerId":id_test, "Survived": pred_labels_test})

    logger.info("results size : {}".format(len(results_df)))
    logger.info("write Results to {}".format(results_path))

    results_df.to_csv(results_path, index=False)
