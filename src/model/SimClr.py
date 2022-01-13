# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

import model.network as models


class SimClr_Model(nn.Module):
    def __init__(self, args, momentum=0.999):

        super(SimClr_Model, self).__init__()

        # self.momentum = momentum

        # Load model
        self.encoder_q = getattr(models, args.model)(
            args, num_classes=128)  # Query Encoder

        # self.encoder_k = getattr(models, args.model)(
        #     args, num_classes=128)  # Key Encoder

        # Add the mlp head
        self.encoder_q.fc = projection_MLP(args)
        # self.encoder_k.fc = models.projection_MLP(args)

    def forward(self, x_q, x_k):

        batch_size = x_q.size(0)

        # Feature representations of the query view from the query encoder
        feat_q = self.encoder_q(x_q)
        feat_k = self.encoder_q(x_k)

        # Compute the logits for the InfoNCE contrastive loss.
        logit = NT_XentLoss(feat_q, feat_k)
        # Update the queue/memory with the current key_encoder minibatch.
        return logit

class projection_MLP(nn.Module):
    def __init__(self, args):
        '''Projection head for the pretraining of the resnet encoder.

            - Uses the dataset and model size to determine encoder output
                representation dimension.
            - Outputs to a dimension of 128, and uses non-linear activation
                as described in SimCLR paper: https://arxiv.org/pdf/2002.05709.pdf
        '''
        super(projection_MLP, self).__init__()

        if args.model == 'resnet18' or args.model == 'resnet34':
            n_channels = 512
        elif args.model == 'resnet50' or args.model == 'resnet101' or args.model == 'resnet152':
            n_channels = 2048
        else:
            raise NotImplementedError('model not supported: {}'.format(args.model))

        self.projection_head = nn.Sequential()

        self.projection_head.add_module('W1', nn.Linear(
            n_channels, n_channels))
        # change
        # self.projection_head.add_module('BN', nn.BatchNorm1d(n_channels))
        self.projection_head.add_module('ReLU', nn.ReLU())
        self.projection_head.add_module('W2', nn.Linear(
            n_channels, 128))

    def forward(self, x):
        return self.projection_head(x)

def NT_XentLoss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape
    device = z1.device
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]

    negatives = similarity_matrix[~diag].view(2*N, -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature

    labels = torch.zeros(2*N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)



