#Filename:	nn.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sel 08 Des 2020 09:44:49  WIB

import torch
import os
import numpy as np
from model.base_model import BaseModel

class NNModel(BaseModel):

    def __init__(self, model_path):

        super().__init__(model_path)
        self.load_model()
        # work in CPU mode
        self.model = self.model.cpu()
        self.model.eval()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = torch.load(self.model_path)
            #self.model = torch.load(self.model_path, map_location = {"cuda:0":"cuda:1"}) 

    def predict_ndarray(self, _input):
        """
        _input: input numpy array
        predict: list of integer
        """
        with torch.no_grad():
            input_tensor = torch.from_numpy(_input)
            pred = self.model(input_tensor).squeeze()
            out = torch.round(pred) 
            return out.numpy().astype(np.int), pred.numpy()

    def predict_tensor(self, _input):
        """
        _input: input tensor
        predict: list of  tensor
        """
        pred = self.model(_input).squeeze()
        out = torch.round(pred)
        return out, pred

    def gradient(self, _input):
        raise NotImplementedError

