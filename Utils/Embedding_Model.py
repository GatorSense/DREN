## PyTorch dependencies
import torch.nn as nn


class Embedding_model(nn.Module):
    
    def __init__(self,model,input_feat_size=524,embed_dim=2):
        
        #inherit nn.module
        super(Embedding_model,self).__init__()
        self.embed_dim = embed_dim
  
        #Define fully connected layer, backbone (feature extractor),
        #and embedding
        # self.fc = model.fc
        self.fc = nn.Linear(embed_dim,model.fc.out_features)
        model.fc = nn.Sequential()
        self.features = model
        self.encoder = nn.Sequential(nn.Linear(input_feat_size,128),nn.ReLU(),
                                     nn.Linear(128,64), nn.ReLU(),
                                     nn.Linear(64,32), nn.ReLU(),
                                     nn.Linear(32,self.embed_dim))
    def forward(self,x):

        #Pass in input through backbone
        x = self.features(x)
        
        # #Pass through fully conneted layer and embedding model (separate)
        # x_fc = self.fc(x)
        # x_embed = self.encoder(x)
        
        #Pass through fully conneted layer and embedding model (in-line)
        x_embed = self.encoder(x)
        x_fc = self.fc(x_embed)
            
        #Return output layer, embedding, and features
        return x_fc, x_embed, x
        
        
        
        
        
        