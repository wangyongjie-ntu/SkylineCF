#Filename:	base_model.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 07 Des 2020 04:16:43  WIB

class BaseModel(object):

    def __init__(self, model_path):

        self.model_path = model_path

    def load_model(self):

        raise NotImplementedError

    def predict(self):

        raise NotImplementedError

    def gradient(self):

        raise NotImplementedError
