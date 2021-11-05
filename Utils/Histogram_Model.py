## PyTorch dependencies
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch

import pdb


class HistRes(nn.Module):
    
    def __init__(self,histogram_layer,parallel=True,model_name ='resnet18',
                 add_bn=True,scale=5,pretrained=True):
        
        #inherit nn.module
        super(HistRes,self).__init__()
        self.parallel = parallel
        self.add_bn = add_bn
        self.scale = scale
        #Default to use resnet18, otherwise use Resnet50
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_ftrs = self.backbone.fc.in_features
            
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_ftrs = self.backbone.fc.in_features
                
        elif model_name == "resnet50_wide":
            self.backbone = models.wide_resnet50_2(pretrained=pretrained)
            num_ftrs = self.backbone.fc.in_features
           
            
        elif model_name == "resnet50_next":
            self.backbone = models.resnext50_32x4d(pretrained=pretrained)
            num_ftrs = self.backbone.fc.in_features
            
        elif model_name == "densenet121":
            model_ft = models.densenet121(pretrained=pretrained,memory_efficient=True)
            num_ftrs = model_ft.classifier.in_features
            
        elif model_name == "efficientnet":
            model_ft = models.efficientnet_b0(pretrained=pretrained)
            num_ftrs = model_ft.classifier.in_features
            
        elif model_name == "regnet":
            model_ft = models.regnet_x_400mf(pretrained)
            num_ftrs = model_ft.classifier.in_features
        
            
        else: 
            print('Model not defined')
            
        if self.add_bn:
            self.bn_norm = nn.BatchNorm2d(num_ftrs)
            
        
        #Define histogram layer and fc
        self.histogram_layer = histogram_layer
        
        try:
            self.fc = self.backbone.fc
            self.backbone.fc = torch.nn.Sequential()
        except:
            self.fc = self.backbone.classifier
            self.backbone.classifier = torch.nn.Sequential()
        
        
    def forward(self,x):

        #All scales except for scale 5 default to parallel
        pdb.set_trace()
        if self.scale == 1:
            output = self.forward_scale_1(x)
        elif self.scale == 2:
            output = self.forward_scale_2(x)
        elif self.scale == 3:
            output = self.forward_scale_3(x)
        elif self.scale == 4: 
            output = self.forward_scale_4(x)
        else: #Default to have histogram layer at end
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)
        
            #Pass through histogram layer and pooling layer
            if(self.parallel):
                if self.add_bn:
                    x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)),start_dim=1)
                else:
                    x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)
        
                x_hist = torch.flatten(self.histogram_layer(x),start_dim=1)
                x_combine = torch.cat((x_pool,x_hist),dim=1)
                output = self.fc(x_combine)
            else:
                x = torch.flatten(self.histogram_layer(x),start_dim=1)
                output = self.fc(x)
     
        return output
    
    def forward_scale_1(self,x):
        
        x = self.backbone.conv1(x)
        x_hist = torch.flatten(self.histogram_layer(x),start_dim=1)    
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
    
        #Pass through histogram layer and pooling layer
        if self.add_bn:
            x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)),start_dim=1)
        else:
            x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)
        x_combine = torch.cat((x_pool,x_hist),dim=1)
        output = self.fc(x_combine)
        
        return output
        
    def forward_scale_2(self,x):
        
        x = self.backbone.conv1(x)    
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x_hist = torch.flatten(self.histogram_layer(x),start_dim=1)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
    
        #Pass through histogram layer and pooling layer
        if self.add_bn:
            x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)),start_dim=1)
        else:
            x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)
        x_combine = torch.cat((x_pool,x_hist),dim=1)
        output = self.fc(x_combine)
        
        return output
        
    def forward_scale_3(self,x):
        
        x = self.backbone.conv1(x)    
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x_hist = torch.flatten(self.histogram_layer(x),start_dim=1)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
    
        #Pass through histogram layer and pooling layer
        if self.add_bn:
            x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)),start_dim=1)
        else:
            x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)
        x_combine = torch.cat((x_pool,x_hist),dim=1)
        output = self.fc(x_combine)
        
        return output
        
    def forward_scale_4(self,x):
        
        x = self.backbone.conv1(x)    
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x_hist = torch.flatten(self.histogram_layer(x),start_dim=1)
        x = self.backbone.layer4(x)
    
        #Pass through histogram layer and pooling layer
        if self.add_bn:
            x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)),start_dim=1)
        else:
            x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)
        x_combine = torch.cat((x_pool,x_hist),dim=1)
        output = self.fc(x_combine)
        
        return output
        
        
        
        
        
        
        
        