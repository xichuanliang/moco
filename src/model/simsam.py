# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import model.network as models


class SimSam_Model(nn.Module):
    def __init__(self, args, momentum=0.999):

        super(SimSam_Model, self).__init__()

        # self.momentum = momentum

        # Load model
        self.encoder_q = getattr(models, args.model)(
            args, num_classes=128)  # Query Encoder

        # self.encoder_k = getattr(models, args.model)(
        #     args, num_classes=128)  # Key Encoder

        self.prodictor = projection_MLP(in_dim = 128)

        # Add the mlp head
        # self.encoder_q.fc = models.projection_MLP(args)
        # self.encoder_k.fc = models.projection_MLP(args)

    def D(self, p, z):  # negative cosine similarity
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    def forward(self, x_q, x_k):

        batch_size = x_q.size(0)

        # Feature representations of the query view from the query encoder
        feat_q = self.encoder_q(x_q)
        feat_q_simsam1 = self.prodictor(feat_q)
        feat_k = self.encoder_q(x_k)
        fea_k_simsam2 = self.prodictor(feat_k)

        # Compute the logits for the InfoNCE contrastive loss.
        logit = self.D(feat_q_simsam1, feat_k) /2 + self.D(fea_k_simsam2, feat_q) / 2
        # Update the queue/memory with the current key_encoder minibatch.
        return logit

class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x


