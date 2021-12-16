import torch
import torch.nn as nn
import torchvision.models as models


class FrontModel(nn.Module):
    def __init__(self):
        super(FrontModel, self).__init__()
        model = models.alexnet()        
        self.features = model.features
       

    def forward(self,x):          
        y = self.features(x) 
        y = y.view(y.size(0), 9216)
        
        return y


class Classifier(nn.Module):
    def __init__(self, num_classes, class_groups, class_groups_idx, index_array):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.class_groups = class_groups
        self.class_groups_idx = class_groups_idx
        self.num_class_groups = len(self.class_groups)
        self.index_array = index_array

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_classes),
        )
   
    def forward(self, x):
        num_obj = list(x.size())[0]
        num_views = list(x.size())[1]
        num_features = list(x.size())[2]
        
        x = x.reshape(num_obj*num_views,num_features)
        y = self.classifier(x)

        y = y.reshape(num_obj,num_views, self.num_classes)     
        y = torch.mean(y,dim=1)

        return y

    def sort_tensor(self,y):
        y_sorted = y[:,0].unsqueeze(1)
        
        for i in range(1,self.num_classes):
            y_sorted = torch.cat((y_sorted,y[:,self.index_array[i]].unsqueeze(1)),dim=1) 
            
        return y_sorted


    def get_mc_output(self,y, class_groups_idx):
            y_mc = self.get_mc_maxima(y,class_groups_idx["0"]).unsqueeze(1)

            for i in range(1,len(class_groups_idx.keys())):
                    y_mc = torch.cat((y_mc,self.get_mc_maxima(y,class_groups_idx[str(i)]).unsqueeze(1)),dim=1)

            return y_mc

    
    def get_mc_maxima(self,y,mc_classes):
       
        y_mc =  y[:,mc_classes[0]].unsqueeze(1)
        
        for i in range(1,len(mc_classes)):
                y_mc = torch.cat((y_mc, y[:,mc_classes[i]].unsqueeze(1)),dim=1)

        y_mc, _ = torch.max(y_mc, dim=1) 

        return y_mc





