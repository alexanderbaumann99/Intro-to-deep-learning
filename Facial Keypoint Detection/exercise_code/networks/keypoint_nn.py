"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np

class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""
    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        #KeypointModel, self
        super().__init__()
        self.hparams = hparams

        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################

        self.pool=nn.MaxPool2d(2,2)
        self.conv1=nn.Conv2d(1,4,hparams['kernel'])
        self.cbn1=nn.BatchNorm2d(4)

        self.conv2=nn.Conv2d(4,8,hparams['kernel'])
        self.cbn2=nn.BatchNorm2d(8)

        self.conv3=nn.Conv2d(8,16,hparams['kernel'])
        self.cbn3=nn.BatchNorm2d(16)

        self.fc1=nn.Linear(1024,hparams['hidden_size'])
        self.fcbn1=nn.BatchNorm1d(hparams['hidden_size'])

        self.fc2=nn.Linear(hparams['hidden_size'],hparams['hidden_size'])
        self.fcbn2=nn.BatchNorm1d(hparams['hidden_size'])

        self.fc3=nn.Linear(hparams['hidden_size'],30)

        self.dp=nn.Dropout(0.4)

        self.act=nn.LeakyReLU()




        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################

        x=x.view((-1,1,96,96))
        x=self.pool(F.relu(self.cbn1(self.conv1(x))))

        x=self.pool(F.relu(self.cbn2(self.conv2(x))))
        x=self.pool(F.relu(self.cbn3(self.conv3(x))))
     
        x=x.view((x.shape[0],-1))

        x=self.dp(self.act(self.fcbn1(self.fc1(x))))
        x=self.dp(self.act(self.fcbn2(self.fc2(x))))
        x=self.fc3(x)
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x

    def general_step(self, batch, batch_idx, mode):
        images, targets = batch["image"], batch["keypoints"]
    
    
        # forward pass
        out = self.forward(images).view(-1,15,2)

        # loss
        loss = F.mse_loss(out, targets)

        return loss

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        
        return avg_loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        logs={'train_loss':loss}
        self.logger.experiment.add_scalar("Loss/Train",loss)

        return {'loss': loss, 'log': logs}
    
    def validation_step(self, batch, batch_idx):
        loss= self.general_step(batch, batch_idx, "val")
        #self.logger.experiment.add_scalar("Loss/Val",loss)
        logs={'val_loss':loss}
        return {'val_loss': loss,'log': logs}

    
    def configure_optimizers(self):

        optim = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################
        optim=torch.optim.Adam(self.parameters(),self.hparams['learning_rate'])
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return optim

    


class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""
    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)
