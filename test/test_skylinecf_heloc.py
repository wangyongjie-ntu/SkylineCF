#Filename:	test_skylinecf_heloc.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Min 03 Jan 2021 04:09:33  WIB

from model.nn import NNModel
from cf.skylineCF_R import skylineCF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.Dataset import Dataset
from utils.helper import load_heloc
import numpy as np
from sklearn.ensemble import IsolationForest

if __name__ == "__main__":

    heloc_df = load_heloc("data/heloc/heloc_dataset_v1.csv")
    d = Dataset(dataframe = heloc_df, continuous_features = 'all', outcome_name = 'RiskPerformance', scaler = MinMaxScaler())
    '''
    features_to_vary = ['MSinceMostRecentTradeOpen', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 
            'MSinceMostRecentDelq', 'MSinceMostRecentInqexcl7days',
            'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']
    '''
    features_to_vary = ['MSinceMostRecentTradeOpen', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']
    clf = NNModel(model_path = 'weights/heloc.pth')
    cf = skylineCF(d, clf, IsolationForest())

    with open("heloc_index.npy", "rb") as f:
        query_index = np.load(f)

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
            "NumBank2NatlTradesWHighUtilization":0
            }
    categorical_rules = {}

    heloc_skylinecf_scores = np.zeros((len(query_index), 10, 3))
    length = np.zeros(len(query_index))

    for i in range(len(query_index)):
        print(i)
        iloc = query_index[i]
        test_instance = d.test_x[iloc:iloc+1]
        results, probs = cf.generate_counterfactuals(test_instance, features_to_vary = features_to_vary)
        if results is None:
            continue

        #results, probs = cf.rank_by_sparsity(results, test_instance, probs, topk = 10)
        skyline_results = d.onehot_decode(results).round(2)
        source = d.onehot_decode(test_instance).round(2)
        tmp = np.zeros((3, len(results)))

        for j in range(len(results)):
            single_result = results[j:j+1]
            score1 = d.compute_continuous_percentile_shift(test_instance, single_result, features_to_vary, normalized = False, method = "sum")
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
        heloc_skylinecf_scores[i, 0:tmp_result, 0] = tmp[0, index[0, 0:tmp_result]]
        # sparsity, larger is better
        iii = index[1, -tmp_result:][::-1]
        heloc_skylinecf_scores[i, 0:tmp_result, 1] = tmp[1, iii]
        # rule-based score, larger is better
        iii = index[2, -tmp_result:][::-1]
        heloc_skylinecf_scores[i, 0:tmp_result, 2] = tmp[2, iii]

    with open("exp_record/heloc_skylinecf_scores.npy", "wb") as f:
        np.save(f, heloc_skylinecf_scores)

    with open("exp_record/heloc_skylinecf_length.npy", "wb") as f:
        np.save(f, length)
