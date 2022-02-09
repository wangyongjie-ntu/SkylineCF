#Filename:	test_skylinecf_heloc.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Min 03 Jan 2021 04:09:33  WIB

from model.nn import NNModel
from cf.skylineCF import skylineCF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.Dataset import Dataset
from utils.helper import load_adult_income
import numpy as np
from sklearn.ensemble import IsolationForest

if __name__ == "__main__":

    adult_df = load_adult_income("data/adult/adult.csv")
    d = Dataset(dataframe = adult_df, continuous_features = ['age', 'education', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'], outcome_name = 'income', scaler = MinMaxScaler())
    features_to_vary = ['capital-gain', 'capital-loss', 'hours-per-week', 'workclass', 'occupation']
    clf = NNModel(model_path = 'weights/adult.pth')
    cf = skylineCF(d, clf, IsolationForest(contamination = 0.01, random_state = 0))
    with open("adult_index.npy", "rb") as f:
        query_index = np.load(f)

    continous_rules = {"capital-gain":1, "capital-loss":0}
    categorical_rules = {"occupation": {"Service":0, "Admin":1, "Blue-Collar":2, "Sales":3, "Other":4, "Military":5, "Professional":6, "White-Collar":7},
            "workclass": {"Other/Unknown":0, "Private":1, "Government":2, "Self-Employed":3}}
    adult_skylinecf_scores = np.zeros((1000, 10, 3))
    length = np.zeros(1000)

    for i in range(1000):
        print(i)
        iloc = query_index[i]
        test_instance = d.test_x[iloc:iloc+1]
        results, probs = cf.generate_counterfactuals(test_instance, features_to_vary = features_to_vary)
        if results is None:
            continue
        #results, probs = cf.rank_by_sparsity(results, test_instance, probs, topk = 10)
        skyline_results = d.onehot_decode(results)
        source = d.onehot_decode(test_instance)
        tmp = np.zeros((3, len(results)))

        for j in range(len(results)):
            single_result = results[j:j+1]
            score1 = d.compute_continuous_percentile_shift(test_instance, single_result, features_to_vary, normalized = False, method = "sum")
            #score2 = d.compute_categorical_changes(test_instance, single_result)
            score2 = d.compute_sparsity(source, skyline_results.iloc[j], features_to_vary)
            score3 = d.compute_actionability_score(source, skyline_results.iloc[j], features_to_vary, continous_rules, categorical_rules)
            tmp[0][j] = score1
            tmp[1][j] = score2
            tmp[2][j] = score3
            #tmp[3][j] = score4
        
        if len(results) > 10:
            tmp_result = 10
        else:
            tmp_result = len(results)
        
        length[i] = tmp_result
        index = tmp.argsort(1)
        # percentile shift, min is better
        adult_skylinecf_scores[i, 0:tmp_result, 0] = tmp[0, index[0, 0:tmp_result]]
        # categorical change, min is better
        #adult_skylinecf_scores[i, 0:tmp_result, 1] = tmp[1, index[1, 0:tmp_result]]
        # sparsity, max is better
        iii = index[1, -tmp_result:][::-1]
        adult_skylinecf_scores[i, 0:tmp_result, 1] = tmp[1, iii]
        # rule-based score, max is better
        iii = index[2, -tmp_result:][::-1]
        adult_skylinecf_scores[i, 0:tmp_result, 2] = tmp[2, iii]

    with open("exp_record/adult_skylinecf_scores.npy", "wb") as f:
        np.save(f, adult_skylinecf_scores)

    with open("exp_record/adult_skylinecf_length.npy", "wb") as f:
        np.save(f, length)
