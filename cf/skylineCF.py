#Filename:	skyline_CF.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 14 Des 2020 03:52:26  WIB

import torch
import numpy as np
import time
from cf.baseCF import ExplainerBase

class skylineCF(ExplainerBase):

    def __init__(self, data_interface, model_interface, anomaly_model):

        super().__init__(data_interface, model_interface)

        self.anomaly = anomaly_model
        self.anomaly.fit(self.data_interface.train_x)

    def generate_counterfactuals(self, query_instance, features_to_vary, target = 0.7):
        
        start_time = time.time()
        if isinstance(query_instance, dict):
            query_instance = self.data_interface.prepare_query(query_instance, normalized = False)

        indices_features_to_vary = self.data_interface.get_indices_of_features_to_vary(features_to_vary)
        y_pred = self.model_interface.predict_ndarray(self.data_interface.train_scaled_x)[0]
        index = np.argwhere(y_pred == 1)
        positive_sample = self.data_interface.train_x[index.squeeze()]
        indices_immutable = list(set(range(len(self.data_interface.onehot_encoded_names))).difference(indices_features_to_vary))
        indices_immutable = np.array(indices_immutable)
        indices_immutable = indices_immutable[np.newaxis, :]
        
        anchor_set = self.find_anchor_set(query_instance, indices_immutable, positive_sample, self.model_interface, self.anomaly, target)
        if len(anchor_set) == 0:
            return None, None

        skyline_set = abs(anchor_set - query_instance)
        skyline_set = np.round(skyline_set, 3)
        skyline_index = self.find_skyline_bnl(skyline_set, to_min = indices_features_to_vary, to_max = [])
        skyline = anchor_set[skyline_index]
        counterfactuals, final_preds = self.find_counterfactual_explanations(skyline, query_instance, self.model_interface, self.anomaly)

        for i in range(len(self.data_interface.encoded_categorical_feature_indices)):
            counterfactuals[:, self.data_interface.encoded_categorical_feature_indices[i]] = np.rint(counterfactuals[:, self.data_interface.encoded_categorical_feature_indices[i]])

        tmp = abs(counterfactuals - query_instance)
        tmp = np.round(tmp, 3)
        skyline_index = self.find_skyline_bnl(tmp, to_min = indices_features_to_vary, to_max = [])
        final_counterfactuals = counterfactuals[skyline_index]
        final_preds = final_preds[skyline_index]

        return final_counterfactuals, final_preds

    def find_anchor_set(self, instance, immutable_index, positive_sample, clf, anomaly, target = 0.7):

        instances = np.repeat(instance, len(positive_sample), axis = 0)
        immutable_features = np.take_along_axis(instances, immutable_index, 1)
        np.put_along_axis(positive_sample, immutable_index, immutable_features, 1)
        positive_sample_normalized = self.data_interface.normalize_data(positive_sample)
        _, pred1 = clf.predict_ndarray(positive_sample_normalized)
        pred1 = (pred1 > target)
        pred2 = anomaly.predict(positive_sample)
        pred = pred1 + pred2
        idx = np.where(pred == 2)
        return positive_sample[idx]
    
    def count_diffs(self, a, b, to_min, to_max):
        n_better = 0
        n_worse = 0

        for f in to_min:
            n_better += a[f] < b[f]
            n_worse += a[f] > b[f]

        for f in to_max:
            n_better += a[f] > b[f]
            n_worse += a[f] < b[f]

        return n_better, n_worse

    def find_skyline_bnl(self, data, to_min, to_max):
        """
        Case 1: The point is dominated by one of the elements in the skyline
        Case 2: The point dominate one or more points in the skyline
        Case 3: The point is same to one of the elements in the skyline
        Case 4: The point is neither better nor worse than all of the points in the skyline
        """
        skyline_index = {0}

        for i in range(1, len(data)):

            to_drop = set()
            is_dominated = False

            for j in skyline_index:
                n_better, n_worse = self.count_diffs(data[i], data[j], to_min, to_max)
                # Case 1
                if n_worse > 0 and n_better == 0:
                    is_dominated = True
                    break

                # Case 2
                if n_better > 0 and n_worse == 0:
                    to_drop.add(j)

                # Case 3
                if n_better == 0 and n_worse == 0:
                    to_drop.add(j)

            if is_dominated:
                continue

            skyline_index = skyline_index.difference(to_drop)
            skyline_index.add(i)

        skyline_index = list(skyline_index)
        return skyline_index

    def find_counterfactual_explanations(self, skyline, query_instance, clf, anomaly, target = 0.7, step = 21):

        counterfactuals = None
        preds = []
        count = 0
        tmp = np.zeros_like(skyline).astype(float)

        for alpha in np.linspace(0, 1, 21):
            x_a = alpha * query_instance + (1 - alpha) * skyline
            prediction = clf.predict_ndarray(self.data_interface.normalize_data(x_a))[1]
            if np.all(prediction < target):
                break
            anomaly_score = anomaly.predict(x_a)
            indicator = (prediction > target) & (anomaly_score == 1)
            idx = np.argwhere(indicator == True).squeeze()
            sample = x_a[idx]
            tmp[idx] = sample
        
        idx = np.argwhere(np.sum(tmp, 1) != 0).squeeze()
        counterfactuals = tmp[idx]
        if len(counterfactuals.shape) == 1:
            counterfactuals = counterfactuals[np.newaxis, :]

        probs = clf.predict_ndarray(self.data_interface.normalize_data(counterfactuals))[1]
        if len(probs.shape) == 0:
            probs = np.array([probs])
        return counterfactuals, probs

    def rank_by_sparsity(self, counterfactuals, query_instance, probs, topk = 4, eps = 1e-2):
        
        diff = np.abs(counterfactuals - query_instance)
        continuous_diff = diff[:, 0:len(self.data_interface.continuous_features_names)]
        diff_ = np.where(continuous_diff < eps, np.zeros_like(continuous_diff), continuous_diff)
        num_of_zeros = (diff_ == 0).sum(1)

        categorical_value = np.zeros((len(counterfactuals), len(self.data_interface.encoded_categorical_feature_indices)))
        categorical_query = np.zeros((1, len(self.data_interface.encoded_categorical_feature_indices)))

        for i in range(len(self.data_interface.encoded_categorical_feature_indices)):
            sub_column = self.data_interface.encoded_categorical_feature_indices[i]
            categorical_value[:, i] = np.argmax(counterfactuals[:, sub_column], axis = 1)
            categorical_query[:, i] = np.argmax(query_instance[:, sub_column], axis = 1)
        
        num_of_zeros += ((categorical_value - categorical_query) == 0).sum(1)
        idx = np.argsort(num_of_zeros)[::-1]
        if len(idx) > topk:
            return counterfactuals[idx[0:topk]], probs[idx[0:topk]]
        else:
            return counterfactuals[idx], probs[idx]
    
    '''
    def rank_by_sparsity(self, counterfactuals, query_instance, probs, topk = 4, eps = 1e-3):

        diff = np.abs(counterfactuals - query_instance)
        diff_ = np.where(diff < eps, np.zeros_like(diff), diff)
        num_of_zeros = (diff_ == 0).sum(1)
        idx = np.argsort(num_of_zeros)[::-1]
        if len(idx) > topk:
            return counterfactuals[idx[0:topk]], probs[idx[0:topk]]
        else:
            return counterfactuals[idx], probs[idx]
        
    def find_counterfactual_explanations_v1(self, skyline, instance, clf, anomaly, target = 0.7, step = 21):

        counterfactuals = None
        preds = []
        count = 0
        for i in range(len(skyline)):
            x_ = skyline[i:i+1]
            for alpha in np.linspace(0, 1, 21):
                x_a = (1-alpha) * instance + alpha * x_
                prediction = clf.predict_ndarray(self.data_interface.normalize_data(x_a))[1]
                if (prediction > target) & (anomaly.predict(x_a) == 1):
                    preds.append(prediction.item())
                    if type(counterfactuals) == type(None):
                        counterfactuals = x_a
                    else:
                        counterfactuals = np.vstack((counterfactuals, x_a))
                    break
                if alpha == 1:
                    count += 1

        return counterfactuals, np.array(preds)
    '''
