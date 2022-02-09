#Filename:	test_dice_heloc.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Min 03 Jan 2021 07:43:01  WIB

import dice_ml
import torch
import numpy as np
from utils.helper import load_heloc
from utils.nondominant_ratio import *
import sys
import os

if __name__ == "__main__":

    set_num = int(sys.argv[1])
    heloc_df = load_heloc("data/heloc/heloc_dataset_v1.csv")
    continuous_features = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 
            'MSinceMostRecentDelq', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days',
            'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']
    '''
    features_to_vary = ['MSinceMostRecentTradeOpen', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 
            'MSinceMostRecentDelq', 'MSinceMostRecentInqexcl7days',
            'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']

    '''
    features_to_vary = ['MSinceMostRecentTradeOpen', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']

    d = dice_ml.Data(dataframe = heloc_df, continuous_features = continuous_features, outcome_name = "RiskPerformance")
    indices_features_to_vary =  d.get_indexes_of_features_to_vary(features_to_vary)
    with open('heloc_index.npy', 'rb') as f:
        index = np.load(f)
    backend = "PYT"
    ml_modelpath = "weights/heloc.pth"
    m = dice_ml.Model(model_path = ml_modelpath, backend = backend)
    exp = dice_ml.Dice(d, m)

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

    score_list = []
    heloc_dice_scores = np.zeros((len(index), set_num, 4))
    saved_name = os.path.join("exp_record/", "heloc_dice_score_{}.npy".format(set_num))


    for i in range(len(index)):
        print(i)
        iloc = index[i]
        query = d.test_df.iloc[iloc, 1:].to_dict()
        dice_exp = exp.generate_counterfactuals(query, total_CFs = set_num, desired_class = 'opposite', features_to_vary = features_to_vary)
        for j in range(set_num):
            score1 = d.compute_continuous_percentile_shift(dice_exp.test_instance, dice_exp.final_cfs[j], features_to_vary, normalized = True, method = "sum")
            score2 = d.compute_sparsity(dice_exp.org_instance, dice_exp.final_cfs_df.iloc[j], features_to_vary)
            score3 = d.compute_actionability_score(dice_exp.org_instance, dice_exp.final_cfs_df.iloc[j], features_to_vary, continous_rules, categorical_rules)
            heloc_dice_scores[i, j, 0] = score1
            heloc_dice_scores[i, j, 1] = score2
            heloc_dice_scores[i, j, 2] = score3

        if set_num == 1:
            heloc_dice_scores[i, :, 3] = 1
        else:
            test_instance = d.prepare_query_instance(query, encode = True).to_numpy() 
            results = np.zeros((set_num, 23))
            for j in range(set_num):
                results[j] = dice_exp.final_cfs[j][0]

            skyline_set = abs(results - test_instance)
            skyline_set = np.round(skyline_set, 3)
            nr_score = nondominant_ratio_v1(skyline_set, to_min = indices_features_to_vary, to_max = [])
            heloc_dice_scores[i, :, 3] = nr_score

    with open(saved_name, "wb") as f:
        np.save(f, heloc_dice_scores)
