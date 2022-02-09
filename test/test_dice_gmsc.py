#Filename:	test_skylinecf_heloc.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Min 03 Jan 2021 04:09:33  WIB

import numpy as np
import dice_ml
from utils.helper import load_GMSC
from utils.nondominant_ratio import *
import sys
import os

if __name__ == "__main__":

    set_num = int(sys.argv[1])
    gmsc_df, _ = load_GMSC("data/GMSC/cs-training.csv")
    continuous_features = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

    d = dice_ml.Data(dataframe = gmsc_df, continuous_features = continuous_features, outcome_name = "SeriousDlqin2yrs")
    features_to_vary = ['RevolvingUtilizationOfUnsecuredLines', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse']

    indices_features_to_vary =  d.get_indexes_of_features_to_vary(features_to_vary)
    with open("gmsc_index.npy", "rb") as f:
        index = np.load(f)
    backend = "PYT"

    ml_modelpath = "weights/Give_me_some_credit.pth"
    m = dice_ml.Model(model_path = ml_modelpath, backend = backend)
    exp = dice_ml.Dice(d, m)
    #continous_rules = {"MonthlyIncome":1, "NumberOfOpenCreditLinesAndLoans":1,
    continous_rules = {"MonthlyIncome":1, "RevolvingUtilizationOfUnsecuredLines":0, "DebtRatio":0,
            "NumberOfTime30-59DaysPastDueNotWorse":0, "NumberOfTimes90DaysLate":0, "NumberOfTime60-89DaysPastDueNotWorse":0}
    categorical_rules = {}
    gmsc_dice_scores = np.zeros((1000, set_num, 4))
    saved_name = os.path.join("exp_record/", "gmsc_dice_score_{}.npy".format(set_num))

    for i in range(1000):
        print(i)
        iloc = index[i]
        query = d.test_df.iloc[iloc].to_dict()
        dice_exp = exp.generate_counterfactuals(query, total_CFs = set_num, desired_class = 'opposite', features_to_vary = features_to_vary)
        for j in range(set_num):
            score1 = d.compute_continuous_percentile_shift(dice_exp.test_instance, dice_exp.final_cfs[j], features_to_vary, normalized = True, method = "sum")
            score2 = d.compute_sparsity(dice_exp.org_instance, dice_exp.final_cfs_df.iloc[j], features_to_vary)
            score3 = d.compute_actionability_score(dice_exp.org_instance, dice_exp.final_cfs_df.iloc[j], features_to_vary, continous_rules, categorical_rules)
            gmsc_dice_scores[i, j, 0] = score1
            gmsc_dice_scores[i, j, 1] = score2
            gmsc_dice_scores[i, j, 2] = score3

        if set_num == 1:
            gmsc_dice_scores[i, :, 3] = 1
        else:
            test_instance = d.prepare_query_instance(query, encode = True).to_numpy() 
            results = np.zeros((set_num, 10))
            for j in range(set_num):
                results[j] = dice_exp.final_cfs[j][0]

            skyline_set = abs(results - test_instance)
            skyline_set = np.round(skyline_set, 3)
            nr_score = nondominant_ratio_v1(skyline_set, to_min = indices_features_to_vary, to_max = [])
            gmsc_dice_scores[i, :, 3] = nr_score

    with open(saved_name, "wb") as f:
        np.save(f, gmsc_dice_scores)
