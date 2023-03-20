import copy
import os
from urllib.parse import non_hierarchical

import cv2
import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import Normalize, Resize


class GradCam:
    hook_a, hook_g = None, None
    hook_handles = []

    def __init__(self, model, conv_layer):
        self.model = model
        self.hook_handles.append(
            self.model._modules.get(conv_layer).register_forward_hook(self._hook_a)
        )
        self.hook_handles.append(
            self.model._modules.get(conv_layer).register_backward_hook(self._hook_g)
        )

    def _hook_a(self, module, input, output):
        self.hook_a = output

    def clear_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def _hook_g(self, module, grad_in, grad_out):
        self.hook_g = grad_out[0]

    def _backprop(self, score):
        loss = score.sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)

    def _get_weights(self, score):
        self._backprop(score)
        return self.hook_g.mean(axis=(2, 3))

    def __call__(self, score, gaze_distrib, device):
        """
        we first get each logit before softmax, then get its alpha for importance.
        Then, the importance was weighted by the probability that it belongs to this class (1-5), using gaze_distrib.
        """
        batch_size, feature_size = self._get_weights(score[0]).shape
        weights_all = torch.zeros(batch_size, 5, feature_size).to(device)
        for i in range(5):
            weights_all[:, i, :] = torch.einsum('ij, i -> ij', self._get_weights(score[:, i]), gaze_distrib[:, i])
        weights = torch.sum(weights_all, dim=1).to(device)

        cam = (weights.unsqueeze(-1).unsqueeze(-1) * self.hook_a.to(device)).sum(dim=1)
        cam_np = cam
        cam_np = F.relu(cam_np)
        cam_np = Resize((224, 224))(cam_np)
        cam_np = cam_np - torch.min(cam_np)
        if torch.max(cam_np) != 0.0:
            cam_np = cam_np / torch.max(cam_np)
        return cam_np


class VisualSimilarityModel(nn.Module):
    def __init__(self, device=None):
        super(VisualSimilarityModel, self).__init__()
        self.device = device
        self.encoder = models.resnet50(pretrained=True).to(self.device)
        self.classifier = nn.Linear(1000, 5).to(self.device)
        self.grad_cam = GradCam(
            model=self.encoder,
            conv_layer='layer4',
        )
        self.score_prototype = torch.nn.Embedding(5, 1000).to(self.device)

    def forward(self, x, gaze_score_distribution, device):
        encoded = self.encoder(x)

        # initialize embedding for each score (each score 1-5 -> a vector presents its characteristic)
        tmp_index = torch.LongTensor([0, 1, 2, 3, 4]).to(self.device)
        prototype_vec_raw = self.score_prototype(tmp_index)

        # get similarity index between score embedding and real image representation, softmax to get pseudo prob
        encoded_rep = encoded.repeat(5, 1, 1).transpose(0, 1)
        prototype_vec = prototype_vec_raw.repeat(encoded.shape[0], 1, 1)
        sim_logit = torch.nn.functional.cosine_similarity(
            encoded_rep, prototype_vec, dim=-1
        )
        sim_score = torch.nn.functional.softmax(sim_logit, dim=-1)
        score_index = torch.FloatTensor([1.0, 2.0, 3.0, 4.0, 5.0]).to(self.device)
        classified = torch.einsum("ij, j -> i", sim_score, score_index)

        cam = self.grad_cam(sim_logit, gaze_score_distribution, device)

        return classified, cam, sim_score
