#Filename:	anomaly.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 14 Des 2020 04:21:34  WIB

import numpy as np
from model.base_model import BaseModel

class BaseAnomaly(BaseModel):

    def __init__(self, model_path):
        self.model_path = model_path

    def load_model(self):

        self.model = 

    def predict(self, _input):

