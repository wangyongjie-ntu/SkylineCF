#Filename:	test_gs_cf_heloc.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sab 02 Jan 2021 06:49:28  WIB

from model.nn import NNModel
from cf.GrowingSphere import GrowingSphere
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.Dataset import Dataset
from utils.helper import load_heloc
from utils.nondominant_ratio import *
import numpy as np
import torch

if __name__ == "__main__":

    heloc_df = load_heloc("data/heloc/heloc_dataset_v1.csv")
    d = Dataset(dataframe = heloc_df, continuous_features = 'all', outcome_name = 'RiskPerformance', scaler = MinMaxScaler())
    '''
    features_to_vary = ['MSinceMostRecentTradeOpen', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 
            'MSinceMostRecentDelq', 'MSinceMostRecentInqexcl7days',
            'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']
    '''
    features_to_vary = ['MSinceMostRecentTradeOpen', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']
    indices_features_to_vary =  d.get_indices_of_features_to_vary(features_to_vary)
    clf = NNModel(model_path = 'weights/heloc.pth')
    cf = GrowingSphere(d, clf)

    with open("heloc_index.npy", "rb") as f:
        index = np.load(f)

    continous_rules = {"ExternalRiskEstimate":1,
            "MSinceOldestTradeOpen":1,
            "MSinceMostRecentTradeOpen":1,
            "AverageMInFile":1,
            "NumSatisfactoryTrades":1,
            "NumTrades60Ever2DerogPubRec":1,
            "NumTrades90Ever2DerogPubRec":1,
            "PercentTradesNeverDelq":1,
            "MSinceMostRecentDelq":1,
            "MSinceMostRecentInqexcl7days":1,
            "NetFractionRevolvingBurden":0,
            "NetFractionInstallBurden":0,
            "NumBank2NatlTradesWHighUtilization":0,
            "NumTradesOpeninLast12M":0,
            "MSinceMostRecentInqexcl7days":0,
            "NumInqLast6M":0
            }
    categorical_rules = {}

    heloc_gscf_scores = np.zeros((len(index), 10, 4))

    for i in range(len(index)):
        print(i)
        iloc = index[i]
        test_instance = d.test_scaled_x[iloc:iloc+1]
        results = np.zeros((10, 23))
        flag = 0
        for j in range(10):
            tmp = cf.generate_counterfactuals(test_instance, features_to_vary = features_to_vary, eta = 1, observation_n = 500)
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
        gs_cf_results = d.onehot_decode(gs_cf_results).round(2)
        source = d.onehot_decode(d.test_x[iloc:iloc+1]).round(2)
        for j in range(10):
            score1 = d.compute_continuous_percentile_shift(test_instance, results[j:j+1], features_to_vary, normalized = True, method = "sum")
            score2 = d.compute_sparsity(source, gs_cf_results.iloc[j], features_to_vary)
            score3 = d.compute_actionability_score(source, gs_cf_results.iloc[j], features_to_vary, continous_rules, categorical_rules)
            print(score3)
            heloc_gscf_scores[i, j, 0] = score1
            heloc_gscf_scores[i, j, 1] = score2
            heloc_gscf_scores[i, j, 2] = score3
        
        skyline_set = abs(results - test_instance)
        skyline_set = np.round(skyline_set, 3)
        nr_final = nondominant_ratio(skyline_set, to_min = indices_features_to_vary, to_max = [])
        heloc_gscf_scores[i, :, 3] = nr_final

    with open("exp_record/heloc_gscf_scores.npy", "wb") as f:
        np.save(f, heloc_gscf_scores)
