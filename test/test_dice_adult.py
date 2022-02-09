#Filename:	test_skylinecf_heloc.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Min 03 Jan 2021 04:09:33  WIB

from cf.skylineCF import skylineCF
import numpy as np
import dice_ml
from utils.helper import load_adult_income
from utils.nondominant_ratio import *
import sys
import os

if __name__ == "__main__":
    
    set_num = int(sys.argv[1])
    income_df = load_adult_income("data/adult/adult.csv")
    d = dice_ml.Data(dataframe = income_df, continuous_features = ['age', 'education', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'], outcome_name = 'income')
    features_to_vary = ['capital-gain', 'capital-loss', 'hours-per-week', 'workclass', 'occupation']
    indices_features_to_vary =  d.get_indexes_of_features_to_vary(features_to_vary)
    with open('adult_index.npy', 'rb') as f:
        index = np.load(f)
    backend = "PYT"

    ml_modelpath = "weights/adult.pth"
    m = dice_ml.Model(model_path = ml_modelpath, backend = backend)
    exp = dice_ml.Dice(d, m)
    continous_rules = {"capital-gain":1, "capital-loss":0}
    categorical_rules = {"occupation": {"Service":0, "Admin":1, "Blue-Collar":2, "Sales":3, "Other":4, "Military":5, "Professional":6, "White-Collar":7},
            "workclass": {"Other/Unknown":0, "Private":1, "Government":2, "Self-Employed":3}}
    
    adult_dice_scores = np.zeros((1000, set_num, 4))
    saved_name = os.path.join("exp_record/", "adult_dice_score_{}.npy".format(set_num))

    for i in range(1000):
        iloc = index[i]
        query = d.test_df.iloc[iloc, 0:-1].to_dict()
        dice_exp = exp.generate_counterfactuals(query, total_CFs = set_num, desired_class = 'opposite', features_to_vary = features_to_vary)
        print(i)
        for j in range(set_num):
            score1 = d.compute_continuous_percentile_shift(dice_exp.test_instance, dice_exp.final_cfs[j], features_to_vary, normalized = True, method = "sum")
            score2 = d.compute_sparsity(dice_exp.org_instance, dice_exp.final_cfs_df.iloc[j], features_to_vary)
            score3 = d.compute_actionability_score(dice_exp.org_instance, dice_exp.final_cfs_df.iloc[j], features_to_vary, continous_rules, categorical_rules)
            adult_dice_scores[i, j, 0] = score1
            adult_dice_scores[i, j, 1] = score2
            adult_dice_scores[i, j, 2] = score3
    
        
        if set_num == 1:
            adult_dice_scores[i, :, 3] = 1
        else:
            test_instance = d.prepare_query_instance(query, encode = True).to_numpy() 
            results = np.zeros((set_num, 36))
            for j in range(set_num):
                results[j] = dice_exp.final_cfs[j][0]

            skyline_set = abs(results - test_instance)
            skyline_set = np.round(skyline_set, 3)
            nr_score = nondominant_ratio_v1(skyline_set, to_min = indices_features_to_vary, to_max = [])
            adult_dice_scores[i, :, 3] = nr_score

    with open(saved_name, "wb") as f:
        np.save(f, adult_dice_scores)
