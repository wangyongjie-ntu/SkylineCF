#Filename:	test_skylinecf_heloc.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sab 02 Jan 2021 06:49:28  WIB

from model.nn import NNModel
from cf.skylineCF import skylineCF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.Dataset import Dataset
from utils.helper import load_GMSC
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
np.random.seed(0)

if __name__ == "__main__":

    gmsc_df, _ = load_GMSC("data/GMSC/cs-training.csv")
    d = Dataset(dataframe = gmsc_df, continuous_features = 'all', outcome_name = 'SeriousDlqin2yrs', scaler = MinMaxScaler())

    features_to_vary = ['RevolvingUtilizationOfUnsecuredLines', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse']
    clf = NNModel(model_path = 'weights/Give_me_some_credit.pth')
    cf = skylineCF(d, clf, IsolationForest(contamination = 0.01, random_state = 0))
    with open("gmsc_index.npy", "rb") as f:
        query_index = np.load(f)
    #continous_rules = {"MonthlyIncome":1, "NumberOfOpenCreditLinesAndLoans":1,
    continous_rules = {"MonthlyIncome":1, "RevolvingUtilizationOfUnsecuredLines":0, "DebtRatio":0,
            "NumberOfTime30-59DaysPastDueNotWorse":0, "NumberOfTimes90DaysLate":0, "NumberOfTime60-89DaysPastDueNotWorse":0}
    categorical_rules = {}

    gmsc_skylinecf_scores = np.zeros((1000, 10, 3))
    length = np.zeros(1000)

    for i in range(1000):
        print(i)
        iloc = query_index[i]
        test_instance = d.test_x[iloc:iloc+1]
        results, probs = cf.generate_counterfactuals(test_instance, features_to_vary = features_to_vary)
        if results is None:
            continue
        #results, probs = cf.rank_by_sparsity(results, test_instance, probs, topk = 10)
        skyline_results = d.onehot_decode(results).round(2)
        source = d.onehot_decode(test_instance)
        tmp = np.zeros((3, len(results)))

        for j in range(len(results)):
            score1 = d.compute_continuous_percentile_shift(test_instance, results[j:j+1], features_to_vary, normalized = False, method ='sum')
            score2 = d.compute_sparsity(source, skyline_results.iloc[j], features_to_vary)
            score3 = d.compute_actionability_score(source, skyline_results.iloc[j], features_to_vary, continous_rules, categorical_rules)
            tmp[0][j] = score1
            tmp[1][j] = score2
            tmp[2][j] = score3

        if len(results) > 10:
            tmp_result = 10
        else:
            tmp_result = len(results)

        length[i] = tmp_result
        index = tmp.argsort(1)
        # perentile shift, min is better
        gmsc_skylinecf_scores[i, 0:tmp_result, 0] = tmp[0, index[0, 0:tmp_result]]
        # sparsity, larger is better
        iii = index[1, -tmp_result:][::-1]
        gmsc_skylinecf_scores[i, 0:tmp_result, 1] = tmp[1, iii]
        # rule-based score, larger is better
        iii = index[2, -tmp_result:][::-1]
        gmsc_skylinecf_scores[i, 0:tmp_result, 2] = tmp[2, iii]
        
    with open("exp_record/gmsc_skylinecf_scores.npy", "wb") as f:
        np.save(f, gmsc_skylinecf_scores)

    with open("exp_record/gmsc_skylinecf_length.npy", "wb") as f:
        np.save(f, length)
