# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import loss_fn
from model.mlp_head import MLPHead as MLPHead

import model.network as models
from .cutmix import CutMixer


class Densecl_Model(nn.Module):
    def __init__(self, args, queue_size=65536, momentum=0.999, temperature=0.07):
        '''
        MoCoV2 model, taken from: https://github.com/facebookresearch/moco.

        Adapted for use in personal Boilerplate for unsupervised/self-supervised contrastive learning.

        Additionally, too inspiration from: https://github.com/HobbitLong/CMC.

        Args:
            init:
                args (dict): Program arguments/commandline arguments.

                queue_size (int): Length of the queue/memory, number of samples to store in memory. (default: 65536)

                momentum (float): Momentum value for updating the key_encoder. (default: 0.999)

                temperature (float): Temperature used in the InfoNCE / NT_Xent contrastive losses. (default: 0.07)

            forward:
                x_q (Tensor): Reprentation of view intended for the query_encoder.

                x_k (Tensor): Reprentation of view intended for the key_encoder.

        returns:

            logit (Tensor): Positve and negative logits computed as by InfoNCE loss. (bsz, queue_size + 1)

            label (Tensor): Labels of the positve and negative logits to be used in softmax cross entropy. (bsz, 1)

        '''
        super(Densecl_Model, self).__init__()

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.T = 0.07

        assert self.queue_size % args.batch_size == 0  # for simplicity

        # Load model
        self.encoder_q = getattr(models, args.model)(
            args, num_classes=128)  # Query Encoder

        self.encoder_k = getattr(models, args.model)(
            args, num_classes=128)  # Key Encoder
        # common mlp head
        self.prodictor = models.dense_Head(args)

        # Add the mlp head
        self.encoder_q.fc = models.projection_MLP(args)
        self.encoder_k.fc = models.projection_MLP(args)
        
        
        #self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        #self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        # Initialize the key encoder to have the same values as query encoder
        # Do not update the key encoder via gradient
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create the queue to store negative samples
        # self.register_buffer("queue", torch.randn(self.queue_size, 128))

        # Create pointer to store current position in the queue when enqueue and dequeue
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue", torch.randn(128, 65536))
        self.register_buffer("queue_dense", torch.randn(128, 65536))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.queue_dense = nn.functional.normalize(self.queue_dense, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.crit_cls = nn.CrossEntropyLoss()
        self.crit_dense = nn.CrossEntropyLoss()

    @torch.no_grad()
    def momentum_update(self):
        '''
        Update the key_encoder parameters through the momentum update:


        key_params = momentum * key_params + (1 - momentum) * query_params

        '''

        # For each of the parameters in each encoder
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.detach().data * (1. - self.momentum)

    @torch.no_grad()
    def shuffled_idx(self, batch_size):
        '''
        Generation of the shuffled indexes for the implementation of ShuffleBN.

        https://github.com/HobbitLong/CMC.

        args:
            batch_size (Tensor.int()):  Number of samples in a batch

        returns:
            shuffled_idxs (Tensor.long()): A random permutation index order for the shuffling of the current minibatch

            reverse_idxs (Tensor.long()): A reverse of the random permutation index order for the shuffling of the
                                            current minibatch to get back original sample order

        '''

        # Generate shuffled indexes
        shuffled_idxs = torch.randperm(batch_size).long().cuda()

        reverse_idxs = torch.zeros(batch_size).long().cuda()

        value = torch.arange(batch_size).long().cuda()

        reverse_idxs.index_copy_(0, shuffled_idxs, value)

        return shuffled_idxs, reverse_idxs

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, dense_keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        dense_keys = nn.functional.normalize(dense_keys.mean(dim=2), dim=1)
        # dense_keys = concat_all_gather(dense_keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert 65536 % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_dense[:, ptr:ptr + batch_size] = dense_keys.T
        ptr = (ptr + batch_size) % 65536 # move pointer

        self.queue_ptr[0] = ptr

    def InfoNCE_logits(self, f_q, f_k):
        '''
        Compute the similarity logits between positive
         samples and positve to all negatives in the memory.

        args:
            f_q (Tensor): Feature reprentations of the view x_q computed by the query_encoder.

            f_k (Tensor): Feature reprentations of the view x_k computed by the key_encoder.

        returns:
            logit (Tensor): Positve and negative logits computed as by InfoNCE loss. (bsz, queue_size + 1)

            label (Tensor): Labels of the positve and negative logits to be used in softmax cross entropy. (bsz, 1)
        '''

        f_k = f_k.detach()

        # Get queue from register_buffer
        f_mem = self.queue.clone().detach()

        # Normalize the feature representations
        f_q = nn.functional.normalize(f_q, dim=1)
        f_k = nn.functional.normalize(f_k, dim=1)
        f_mem = nn.functional.normalize(f_mem, dim=1)


        # Compute sim between positive views
        pos = torch.bmm(f_q.view(f_q.size(0), 1, -1),
                        f_k.view(f_k.size(0), -1, 1)).squeeze(-1)

        # Compute sim between postive and all negatives in the memory
        neg = torch.mm(f_q, f_mem.transpose(1, 0))
        # neg_change = torch.mm(f_k, f_mem.transpose(1, 0))
        # 3
        # neg3 = 0.5 * torch.mm(f_q, f_mem.transpose(1, 0))
        # neg_change3 = 0.5 * torch.mm(f_k, f_mem.transpose(1, 0))
        # neg = neg3 + neg_change3

        logits = torch.cat((pos, neg), dim=1)
        # logits1 = torch.cat((pos, neg), dim=1)
        # logits = torch.cat((logits1, neg_change), dim=1)
        # logits_change = torch.cat((pos, neg_change), dim=1)

        logits /= self.temperature
        # logits_change /= self.temperature

        # Create labels, first logit is postive, all others are negative
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # return logits, labels , logits_change
        return logits, labels


    def forward(self, x_q, x_k):
        #knn
        # f_mem = self.queue.clone().detach()
        # f_mem = nn.functional.normalize(f_mem, dim=1)

        batch_size = x_q.size(0)

        # Feature representations of the query view from the query encoder
        # feat_q = self.encoder_q(x_q)
        # q = feat_q
        # dense_q = feat_q
        q, feat_q = self.encoder_q(x_q)  # queries: NxC
        dense_q = self.prodictor(feat_q)
        q = nn.functional.normalize(q, dim=1)

        n, c, h, w = feat_q.size()
        dim_dense = dense_q.size(1)
        dense_q, feat_q = dense_q.view(n, dim_dense, -1), feat_q.view(n, c, -1)
        dense_q = nn.functional.normalize(dense_q, dim=1)


        #knn
        # q_queue_knn= torch.mm(feat_q, f_mem.transpose(1, 0))
        # feat_q_knn1 = f_mem[torch.argmax(q_queue_knn,1)]
        # feat_q_byol1 = self.prodictor(feat_q_knn1)
        # feat_a = self.encoder_q(x_a)
        # a_queue_knn = torch.mm(feat_a, f_mem.transpose(1, 0))
        # feat_a_knn1 = f_mem[torch.argmax(a_queue_knn, 1)]
        # feat_a_byol2 = self.prodictor(feat_a_knn1)

        #moco+byol
        # feat_q_byol1 = self.prodictor(feat_q)
        # feat_a_byol2 = self.prodictor(self.encoder_q(x_a))
        # feat_k_byol2 = self.prodictor(self.encoder_q(x_k))

        # feat_q_byol1 = feat_q
        # feat_a_byol2 = self.encoder_q(x_a)

        # TODO: shuffle ids with distributed data parallel
        # Get shuffled and reversed indexes for the current minibatch
        # shuffled_idxs, reverse_idxs = self.shuffled_idx(batch_size)

        with torch.no_grad():
            # Update the key encoder
            self.momentum_update()

            # Shuffle minibatch
            # x_k = x_k[shuffled_idxs]
            # x_a = x_a[shuffled_idxs]
            # x_q = x_q[shuffled_idxs]

            # Feature representations of the shuffled key view from the key encoder
            # feat_k = self.encoder_k(x_k)
            # feat_a_byol1 = self.encoder_k(x_a)
            # feat_q_byol2 = self.encoder_k(x_q)
            # feat_k_byol1 = self.encoder_k(x_k)

            # reverse the shuffled samples to original position
            # feat_k = feat_k[reverse_idxs]
            # feat_a_byol1 = feat_a_byol1[reverse_idxs]
            # feat_q_byol2 = feat_q_byol2[reverse_idxs]


            k, feat_k = self.encoder_k(x_k)  # keys: NxC
            dense_k = self.prodictor(feat_k)
            k = nn.functional.normalize(k, dim=1)
            dense_k, feat_k = dense_k.view(n, dim_dense, -1), feat_k.view(n, c, -1)
            dense_k_norm = nn.functional.normalize(dense_k, dim=1)

            feat_q_norm = nn.functional.normalize(feat_q, dim=1)
            feat_k_norm = nn.functional.normalize(feat_k, dim=1)
            cosine = torch.einsum('nca,ncb->nab', feat_q_norm, feat_k_norm)
            pos_idx = cosine.argmax(dim=-1)
            dense_k_norm = dense_k_norm.gather(2, pos_idx.unsqueeze(1).expand(-1, dim_dense, -1))

        # Compute the logits for the InfoNCE contrastive loss.
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss_cls = self.crit_cls(logits, labels)

        ## densecl logits
        d_pos = torch.einsum('ncm,ncm->nm', dense_q, dense_k_norm).unsqueeze(1)
        d_neg = torch.einsum('ncm,ck->nkm', dense_q, self.queue_dense.clone().detach())
        logits_dense = torch.cat([d_pos, d_neg], dim=1)
        logits_dense = logits_dense / self.T
        labels_dense = torch.zeros((n, h * w), dtype=torch.long).cuda()

        loss_dense = self.crit_dense(logits_dense, labels_dense)

        extra = {'qk': [q, k], 'dense_qk': [dense_q, dense_k_norm],
                 'logits': logits, 'labels': labels}

        ## regionCL
        #if self.cutmixer:
        #    loss_cutmix = self.cutmixer.forward_mix(
        #        self.encoder_q, x_q, q, k, self.queue.detach())
        #    extra.update({'loss_cutmix': loss_cutmix})
        self._dequeue_and_enqueue(k, dense_k)
        # return logit, label, logit_change
        return loss_cls, loss_dense, extra
