#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .parser_utils import load_and_preprocess_data


class ParserModel(nn.Module):
    """
    A feedforward neural network that leverages word embeddings and a single hidden layer
    to predict which transition should be applied in a transition-based parsing pipeline.

    Implementation Notes:
        - "ParserModel" inherits from "nn.Module", the base class for all neural networks in PyTorch.
        - Use the "__init__" method to define layers such as embeddings, linear layers, and dropout.
        - Other methods (e.g., "forward") access the defined layers via the "self." prefix.
        - Refer to PyTorch documentation (https://pytorch.org/docs/stable/nn.html) for extended details.
    """

    def __init__(
        self, embeddings, n_features=36, hidden_size=200, n_classes=3, dropout_prob=0.5
    ):
        """
        Initialize the ParserModel.

        Args:
            embeddings (Tensor): Pretrained word embeddings of shape (num_words, embedding_size).
            n_features (int): Number of input features to be concatenated.
            hidden_size (int): Number of hidden units in the single hidden layer.
            n_classes (int): Number of output classes (predicted transitions).
            dropout_prob (float): Dropout probability applied after the hidden layer.
        """
        super(ParserModel, self).__init__()
        torch.manual_seed(0)
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.pretrained_embeddings = nn.Embedding(embeddings.shape[0], self.embed_size)
        self.pretrained_embeddings.weight = nn.Parameter(torch.tensor(embeddings))

        # Linear layer mapping embeddings to hidden units
        self.embed_to_hidden = nn.Linear(
            self.n_features * self.embed_size, self.hidden_size
        )
        # Dropout layer
        self.dropout = nn.Dropout(self.dropout_prob)
        # Linear layer mapping hidden units to output logits
        self.hidden_to_logits = nn.Linear(self.hidden_size, self.n_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters of the linear layers using Xavier Uniform initialization.
        This approach often leads to better convergence for neural networks.
        """
        # Initialization of the weight matrix of embed_to_hidden with Xavier Uniform
        self.embed_to_hidden.weight = nn.init.xavier_uniform_(
            self.embed_to_hidden.weight, gain=1
        )
        # Initialization of the weight matrix of hidden_to_logits with Xavier Uniform.
        self.hidden_to_logits.weight = nn.init.xavier_uniform_(
            self.hidden_to_logits.weight, gain=1
        )

    def embedding_lookup(self, t):
        """
        Maps input tokens to their corresponding embedding vectors, then flattens them.

        Args:
            t (Tensor): A batch of token indices (batch_size, n_features).

        Returns:
            x (Tensor): Reshaped embeddings of shape (batch_size, n_features * embed_size).
        """

        # Retrieve embeddings using self.pretrained_embeddings.
        embeddings = self.pretrained_embeddings(t)

        # Reshape the output from (batch_size, n_features, embed_size)
        # to (batch_size, n_features * embed_size).
        x = embeddings.view(embeddings.size(0), -1)
        return x

    def forward(self, t):
        """
        Defines the forward pass of the network. It looks up embeddings,
        processes them through a hidden layer with ReLU, applies dropout,
        and finally produces logits via another linear layer.

        Args:
            t (Tensor): A batch of token indices (batch_size, n_features).

        Returns:
            logits (Tensor): Raw output scores (batch_size, n_classes) before softmax.
        """

        # Step 1: Lookup embeddings for input tensor `t`.
        embeddings = self.embedding_lookup(t)

        # Step 2: Apply the linear layer that maps embeddings to hidden layer (`embed_to_hidden`).
        linear_layer = self.embed_to_hidden(embeddings)

        # Step 3: Apply ReLU non-linearity to the output of the linear layer to get hidden units.
        hidden_units = nn.functional.relu(linear_layer)

        # Step 4: Apply dropout regularization to prevent overfitting.
        dropout_layer = self.dropout(hidden_units)

        # Step 5: Apply final linear layer (`hidden_to_logits`) to get the logits (raw scores before softmax).
        logits = self.hidden_to_logits(dropout_layer)

        return logits
