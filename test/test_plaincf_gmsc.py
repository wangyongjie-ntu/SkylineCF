#Filename:	test_plaincf_heloc.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sab 02 Jan 2021 06:49:28  WIB

from model.nn import NNModel
from cf.plainCF import PlainCF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.Dataset import Dataset
from utils.helper import load_GMSC
from utils.nondominant_ratio import *
import numpy as np

if __name__ == "__main__":

    gmsc_df, _ = load_GMSC("data/GMSC/cs-training.csv")
    d = Dataset(dataframe = gmsc_df, continuous_features = 'all', outcome_name = 'SeriousDlqin2yrs', scaler = MinMaxScaler())

    features_to_vary = ['RevolvingUtilizationOfUnsecuredLines', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse']
    indices_features_to_vary =  d.get_indices_of_features_to_vary(features_to_vary)
    clf = NNModel(model_path = 'weights/Give_me_some_credit.pth')
    cf = PlainCF(d, clf)
    with open("gmsc_index.npy", "rb") as f:
        index = np.load(f)

    continous_rules = {"MonthlyIncome":1, "RevolvingUtilizationOfUnsecuredLines":0, "DebtRatio":0,
            "NumberOfTime30-59DaysPastDueNotWorse":0, "NumberOfTimes90DaysLate":0, "NumberOfTime60-89DaysPastDueNotWorse":0}
    categorical_rules = {}

    gmsc_plaincf_scores = np.zeros((1000, 10, 4))

    for i in range(1000):
        print(i)
        iloc = index[i]
        test_instance = d.test_scaled_x[iloc:iloc+1]
        results = np.zeros((10, 10))
        for j in range(10):
            results[j] = cf.generate_counterfactuals(test_instance, features_to_vary = features_to_vary)

        plaincf_results = d.denormalize_data(results)
        plaincf_results = d.onehot_decode(plaincf_results)
        source = d.onehot_decode(d.test_x[iloc:iloc+1])
        for j in range(10):
            score1 = d.compute_continuous_percentile_shift(test_instance, results[j:j+1], features_to_vary, normalized = True, method = "sum")
            score2 = d.compute_sparsity(source, plaincf_results.iloc[j], features_to_vary)
            score3 = d.compute_actionability_score(source, plaincf_results.iloc[j], features_to_vary, continous_rules, categorical_rules)
            gmsc_plaincf_scores[i, j, 0] = score1
            gmsc_plaincf_scores[i, j, 1] = score2
            gmsc_plaincf_scores[i, j, 2] = score3

        skyline_set = abs(results - test_instance)
        skyline_set = np.round(skyline_set, 3)
        nr_final = nondominant_ratio(skyline_set, to_min = indices_features_to_vary, to_max = [])
        gmsc_plaincf_scores[i, :, 3] = nr_final

    with open("exp_record/gmsc_plaincf_scores.npy", "wb") as f:
        np.save(f, gmsc_plaincf_scores)

