#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

sys.path.append("..")
import os
import time
from datetime import datetime

import math
import torch
from torch import nn, optim
from tqdm import tqdm

from .parser_model import ParserModel
from .parser_utils import minibatches, load_and_preprocess_data, AverageMeter


# -----------------
# Primary Functions
# -----------------
def train(
    parser,
    train_data,
    dev_data,
    output_path,
    batch_size=1024,
    n_epochs=10,
    lr=0.0005,
    device=torch.device("cpu"),
):
    """
    Train the neural dependency parser across multiple epochs, saving the model
    with the best development set performance.

    Args:
        parser (Parser): The neural dependency parser to train.
        train_data: The training dataset containing input features and labels.
        dev_data: The development dataset for evaluation during training.
        output_path (str): Path to save the model with the best performance.
        batch_size (int): Number of training examples in each batch.
        n_epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        device (torch.device): Device to run the training on (CPU/GPU).

    Returns:
        best_dev_UAS (float): The highest Unlabeled Attachment Score (UAS) achieved on the dev set.
    """
    best_dev_UAS = 0

    optimizer = optim.Adam(
        parser.model.parameters(),
        lr=lr,
    )
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_UAS = train_for_epoch(
            parser, train_data, dev_data, optimizer, loss_func, batch_size, device
        )
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            print("New best dev UAS! Saving model.")

            torch.save(parser.model.state_dict(), output_path)
        print("")

    return best_dev_UAS


def train_for_epoch(
    parser, train_data, dev_data, optimizer, loss_func, batch_size, device
):
    """
    Perform one epoch of training for the neural dependency parser.

    Args:
        parser (Parser): The neural dependency parser to train.
        train_data: The training dataset containing input features and labels.
        dev_data: The development dataset for evaluation.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        loss_func (nn.CrossEntropyLoss): Loss function for training.
        batch_size (int): Number of examples in each batch.
        device (torch.device): Device to run training on (CPU/GPU).

    Returns:
        dev_UAS (float): Unlabeled Attachment Score (UAS) on the development dataset.
    """
    parser.model.train()  # Set the model to training mode
    n_minibatches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()

    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            optimizer.zero_grad()  # Reset gradients
            loss = 0.0  # Initialize loss for the current batch
            train_x = torch.from_numpy(train_x).long().to(device=device)
            train_y = torch.from_numpy(train_y.nonzero()[1]).long().to(device=device)

            # Forward pass: compute predictions and loss
            logits = parser.model(train_x)
            cross_entropy_loss = loss_func(logits, train_y)
            loss += cross_entropy_loss

            # Backward pass: compute gradients and update parameters
            cross_entropy_loss.backward()
            optimizer.step()

            prog.update(1)
            loss_meter.update(loss.item())

    print("Average Train Loss: {:.4f}".format(loss_meter.avg))

    print("Evaluating on dev set")
    parser.model.eval()  # Set the model to evaluation mode
    dev_UAS, _ = parser.parse(dev_data, device)
    print("- dev UAS: {:.2f}%".format(dev_UAS * 100.0))
    return dev_UAS
