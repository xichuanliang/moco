# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import model.network as models
from model.mlp_head import MLPHead as MLPHead
from utils import loss_fn


class Byol_Model(nn.Module):
    def __init__(self, args, momentum=0.999):
        '''


        '''
        super(Byol_Model, self).__init__()

        self.momentum = momentum

        # Load model
        self.encoder_q = getattr(models, args.model)(
            args, num_classes=128)  # Query Encoder

        self.encoder_k = getattr(models, args.model)(
            args, num_classes=128)  # Key Encoder

        self.prodictor = MLPHead(in_channels = 128)

        # Add the mlp head
        self.encoder_q.fc = models.projection_MLP(args)
        self.encoder_k.fc = models.projection_MLP(args)

        # Initialize the key encoder to have the same values as query encoder
        # Do not update the key encoder via gradient
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.detach().data * (1. - self.momentum)


    def forward(self, x_q, x_k):

        batch_size = x_q.size(0)

        # Feature representations of the query view from the query encoder
        feat_q = self.encoder_q(x_q)
        feat_q_byol1 = self.prodictor(feat_q)
        fea_k_byol2 = self.prodictor(self.encoder_q(x_k))

        # TODO: shuffle ids with distributed data parallel
        # Get shuffled and reversed indexes for the current minibatch
        # shuffled_idxs, reverse_idxs = self.shuffled_idx(batch_size)

        with torch.no_grad():
            # Update the key encoder
            self.momentum_update()

            # Shuffle minibatch
            # x_k = x_k[shuffled_idxs]
            # x_q = x_q[shuffled_idxs]

            # Feature representations of the shuffled key view from the key encoder
            feat_k = self.encoder_k(x_k)
            feat_q_byol2 = self.encoder_k(x_q)

            # reverse the shuffled samples to original position
            # feat_k = feat_k[reverse_idxs]
            # feat_q_byol2 = feat_q_byol2[reverse_idxs]

        # Compute the logits for the InfoNCE contrastive loss.
        logit2 = loss_fn(feat_q_byol1, feat_k)
        logit3 = loss_fn(fea_k_byol2, feat_q_byol2)
        logit = (logit2+logit3).mean()

        # Update the queue/memory with the current key_encoder minibatch.

        return logit
