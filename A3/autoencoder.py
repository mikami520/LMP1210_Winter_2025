#!/usr/bin/env python
# coding=utf-8
"""
Author       : Chris Xiao yl.xiao@mail.utoronto.ca
Date         : 2025-02-15 23:53:25
LastEditors  : Chris Xiao yl.xiao@mail.utoronto.ca
LastEditTime : 2025-02-16 04:37:11
FilePath     : /LMP1210_Winter_2025/A3/autoencoder.py
Description  : autoencoder script
I Love IU
Copyright (c) 2025 by Chris Xiao yl.xiao@mail.utoronto.ca, All Rights Reserved.
"""

import torch
import torch.nn as nn


def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(AutoEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Mish(),
            nn.Linear(512, 256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Linear(128, latent_dim),
            nn.Mish(),
            nn.Linear(latent_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x)


class AutoEncoderTrainer:
    def __init__(self, input_dim, num_epochs, device):
        self.input_dim = input_dim
        self.device = device
        self.model = AutoEncoder(input_dim=input_dim, latent_dim=64).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00005)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )
        self.num_epochs = num_epochs
        self.init_params()

    def train(self, data_loader):
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for data in data_loader:
                self.optimizer.zero_grad()
                data = data.to(self.device)
                latent = self.model(data)
                loss = self.criterion(latent, data)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(data_loader)}")
            self.lr_scheduler.step()

        torch.save(self.model.state_dict(), "autoencoder.pth")
        print("Model Saved as 'autoencoder.pth'")

    def get_latent_representation(self, data, pretrain=None):
        if pretrain is not None:
            ckpt = torch.load(pretrain)
        else:
            ckpt = torch.load("autoencoder.pth")
        self.model.load_state_dict(ckpt)
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            latent = self.model(data)
        return latent

    def init_params(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
