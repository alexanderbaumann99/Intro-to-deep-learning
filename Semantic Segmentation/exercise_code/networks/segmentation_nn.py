"""SegmentationNN"""
import torch
from torch import functional
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import os
import math



class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        self.encoder=models.mobilenet_v2(pretrained=True)
        self.encoder=nn.Sequential(*list(self.encoder.features))

        for layer in self.encoder:
            for params in layer.parameters():
                params.requires_grad=False


        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(1280,512,1),
            nn.ReLU(inplace=True),
            nn.Upsample([120,120]),
            nn.ConvTranspose2d(512,64,3),
            nn.ReLU(inplace=True),
            nn.Upsample([240,240]),
            nn.Conv2d(64,num_classes,3,1,1),
            nn.ReLU(inplace=True)
        )
        
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x=x.view((-1,3,240,240))
        x=self.encoder(x)
        x=self.decoder(x)
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        # forward pass
        out = self.forward(inputs)

        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        # loss
        loss = loss_func(out, targets)

        return {'loss': loss}

    def configure_optimizers(self):

        optim=torch.optim.Adam(self.parameters(),lr=self.hparams['learning_rate'])
        
        return optim
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        # forward pass
        out = self.forward(inputs)

        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        # loss
        loss = loss_func(out, targets)
        #self.logger.experiment.add_scalar("Loss/Val",loss)
        #logs={'val_loss':loss}
        return {'val_loss': loss}

      
    

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
