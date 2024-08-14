import numpy as np
from utils import decode_captions
import progressbar

import torch


class Trainer(object):

    def __init__(self, model, train_dataloader, val_dataloader, learning_rate=0.001, num_epochs=10, print_every=10,
                 verbose=True, device='cuda'):

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.print_every = print_every
        self.verbose = verbose
        self.loss_history = []
        self.val_loss_history = []
        self.device = device
        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)

    def loss(self, predictions, labels):
        # TODO - Compute cross entropy loss between predictions and labels.
        # Make sure to compute this loss only for indices where label is not the null token.
        # The loss should be averaged over batch and sequence dimensions.
        predictions = predictions.view(-1, predictions.size(-1))  # shape (batch_size * seq_length, vocab_size)

        # Reshape labels to (batch_size * seq_length)
        labels = labels.contiguous().view(-1).long()

        criterion = torch.nn.CrossEntropyLoss(ignore_index=3)  # ignore UNK
        # criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(predictions, labels)
        return loss

    def val(self):
        """
        Run validation to compute loss and BLEU-4 score.
        """
        self.model.eval()
        val_loss = 0
        num_batches = 0
        for batch in self.val_dataloader:
            features, captions = batch[0].to(self.device), batch[1].to(self.device)
            logits = self.model(features, captions[:, :-1])

            loss = self.loss(logits, captions[:, 1:])
            val_loss += loss.detach().cpu().numpy()
            num_batches += 1

        self.model.train()
        return val_loss / num_batches

    def train(self):
        """
        Run optimization to train the model.
        """
        for i in range(self.num_epochs):
            epoch_loss = 0
            num_batches = 0
            bar = progressbar.ProgressBar(max_value=len(self.train_dataloader),
                                          widgets=[f'ITERATION {i + 1} ', progressbar.Percentage(), ' ',
                                                   progressbar.Bar(), ' ', progressbar.ETA(), ])
            for ii, batch in enumerate(self.train_dataloader):
                bar.update(ii + 1)
                features, captions = batch[0].to(self.device), batch[1].to(self.device)
                logits = self.model(features, captions[:, :-1])

                loss = self.loss(logits, captions[:, 1:])
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                epoch_loss += loss.detach().cpu().numpy()
                num_batches += 1

            self.loss_history.append(epoch_loss / num_batches)
            if self.verbose and (i + 1) % self.print_every == 0:
                self.val_loss_history.append(self.val())
                print("(epoch %d / %d) loss: %f" % (i + 1, self.num_epochs, self.loss_history[-1]))
