import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=64,            
                kernel_size=(20,2),              
                stride=(1,1),                   
                padding='same',                  
            ),
            nn.ReLU(),                                       
            nn.MaxPool2d(kernel_size=(20, 1), stride=(5, 1)),
            nn.BatchNorm2d(64)    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(64, 64, (10,2), (1,1), 'same'),
            nn.ReLU(),                       
            nn.MaxPool2d((4,1), stride=(2,1), padding = (2,0), ceil_mode = True), #n.b. had to do a bit of fiddling with padding to get sizes right
        )


        # input - 5 features from data_features (col 2 = !col 3)
        # plus number of recordings
        self.fc1 = nn.Linear(6, 12)

        # fully connected layer, output 1 classe
        self.out = nn.Sequential(
            nn.Linear(64 * 6 * 30 + 12, 1024),
            nn.ReLU(),
            nn.Dropout(0.8556),
            nn.Linear(1024, 512),
            nn.ReLU())
            
            
        self.lastlayer = nn.Sequential(
            nn.Dropout(0.8556),
            nn.Linear(512, 1))
        
    def forward(self, x, dems):
        x = self.conv1(x.float())
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)

        # add in the demographics here, and push them through an 'fc1' layer
        # note that we are expecting dems to be 6x1
        dems = self.fc1(dems.float())

        #concatenate demographics with x
        x = torch.cat((dems, x), dim=1)

        # n.b. the input to 'out' needs to be updated to include dem size
        x = self.out(x)
        output = self.lastlayer(x)
        return output, x    # return x for visualization

class CNNEnsemble(nn.Module):
    def __init__(self, model1, model2, model3, model4, model5):
        super(CNNEnsemble, self).__init__()
        self.model1= model1
        self.model2= model2
        self.model3= model3
        self.model4= model4
        self.model5= model5
        
        # Remove last linear layer
        self.model1.lastlayer = nn.Identity()
        self.model2.lastlayer = nn.Identity()
        self.model3.lastlayer = nn.Identity()
        self.model4.lastlayer = nn.Identity()
        self.model5.lastlayer = nn.Identity()
        
        # Create new classifier
        # n.b. this might not be the best way to combine, of course
        self.classifier = nn.Sequential(
            nn.Dropout(0.8556),
            nn.Linear(512 * 5, 512),
            nn.Dropout(0.8556),
            nn.Linear(512, 1))
        
    def forward(self, x, dems):
        x1,_ = self.model1(x.clone(), dems.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2,_ = self.model2(x.clone(), dems.clone())
        x2 = x2.view(x2.size(0), -1)
        x3,_ = self.model2(x.clone(), dems.clone())
        x3 = x3.view(x3.size(0), -1)
        x4,_ = self.model2(x.clone(), dems.clone())
        x4 = x4.view(x4.size(0), -1)
        x5,_ = self.model2(x.clone(), dems.clone())
        x5 = x5.view(x5.size(0), -1)
        out = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.classifier(nn.functional.relu(out))
        return out