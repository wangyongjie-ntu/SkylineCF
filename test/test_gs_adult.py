#Filename:	test_gs_cf_heloc.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sab 02 Jan 2021 06:49:28  WIB

from model.nn import NNModel
from cf.GrowingSphere import GrowingSphere
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.Dataset import Dataset
from utils.helper import load_adult_income
from utils.nondominant_ratio import *
import numpy as np
import torch
np.random.seed(0)

if __name__ == "__main__":

    heloc_df = load_adult_income("data/adult/adult.csv")
    d = Dataset(dataframe = heloc_df, continuous_features = ['age', 'education', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'], outcome_name = 'income', scaler = MinMaxScaler())
    features_to_vary = ['capital-gain', 'capital-loss', 'hours-per-week', 'workclass', 'occupation']
    indices_features_to_vary =  d.get_indices_of_features_to_vary(features_to_vary)
    clf = NNModel(model_path = 'weights/adult.pth')
    cf = GrowingSphere(d, clf)
    with open("adult_index.npy", "rb") as f:
        index = np.load(f)

    continous_rules = {"capital-gain":1, "capital-loss":0}
    categorical_rules = {"occupation": {"Service":0, "Admin":1, "Blue-Collar":2, "Sales":3, "Other":4, "Military":5, "Professional":6, "White-Collar":7},
            "workclass": {"Other/Unknown":0, "Private":1, "Government":2, "Self-Employed":3}}

    adult_gscf_scores = np.zeros((1000, 10, 4))
    for i in range(1000):
        print(i)
        iloc = index[i]
        test_instance = d.test_scaled_x[iloc:iloc+1]
        results = np.zeros((10, 36))
        flag = 0
        for j in range(10):
            tmp = cf.generate_counterfactuals(test_instance, features_to_vary = features_to_vary)
            if len(tmp.shape) == 1:
                length = 1
                results[flag] = tmp
            else:
                length = len(tmp)
                if flag + length < 10:
                    results[flag:flag + length] = tmp
                else:
                    results[flag:10] = tmp[0:10-flag]

            flag += length
            if flag >= 10:
                break

        gs_cf_results = d.denormalize_data(results)
        gs_cf_results = d.onehot_decode(gs_cf_results)
        source = d.onehot_decode(d.test_x[iloc:iloc+1])
        for j in range(10):
            score1 = d.compute_continuous_percentile_shift(test_instance, results[j:j+1], features_to_vary, normalized = True)
            #score2 = d.compute_categorical_changes(test_instance, results[j:j+1])
            score2 = d.compute_sparsity(source, gs_cf_results.iloc[j], features_to_vary)
            score3 = d.compute_actionability_score(source, gs_cf_results.iloc[j], features_to_vary, continous_rules, categorical_rules)
            adult_gscf_scores[i, j, 0] = score1
            adult_gscf_scores[i, j, 1] = score2
            adult_gscf_scores[i, j, 2] = score3

        skyline_set = abs(results - test_instance)
        skyline_set = np.round(skyline_set, 3)
        nr_final = nondominant_ratio(skyline_set, to_min = indices_features_to_vary, to_max = [])
        adult_gscf_scores[i, :, 3] = nr_final
        
    with open("exp_record/adult_gscf_scores.npy", "wb") as f:
        np.save(f, adult_gscf_scores)
