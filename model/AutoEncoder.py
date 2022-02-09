import torch
import torch.nn as nn

class autoencoder(nn.Module):
    def __init__(self, data_interface):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(36,128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(True), 
            nn.Dropout(0.5),
            nn.Linear(64, 12), 
            nn.ReLU(True), 
            nn.Dropout(0.5),
            nn.Linear(12, 10))
        self.decoder = nn.Sequential(
            nn.Linear(10, 12),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Dropout(0.5),
            nn.Linear(128,36))
        self.sig = nn.Sigmoid()
        self.data_interface = data_interface

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        for v in self.data_interface.encoded_categorical_feature_indices:    
            start_index = v[0]
            end_index = v[-1] + 1
            x[:,start_index:end_index] = self.sig(x[:,start_index:end_index])
        return x

    def compute_loss(self, output, target):
        con_criterion = nn.MSELoss()
        cat_criterion = nn.BCELoss()
        
        cat_loss = 0
        con_loss = 0
        
        for v in self.data_interface.encoded_categorical_feature_indices:
            start_index = v[0]
            end_index = v[-1]+1
            cat_loss += cat_criterion(output[:, start_index:end_index], target[:, start_index:end_index])
        
        categorial_indices = []
        for v in self.data_interface.encoded_categorical_feature_indices:
            categorial_indices.extend(v)

        continuous_indices = list(set(range(36)).difference(categorial_indices))
        con_loss = con_criterion(output[:, continuous_indices], target[:, continuous_indices])
        loss = torch.mean(cat_loss + con_loss) 
        return loss
