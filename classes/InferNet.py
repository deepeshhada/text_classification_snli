import torch
import torch.nn as nn


class InferNet(nn.Module):
	def __init__(self, options):
		super(InferNet, self).__init__()
		self.embedding = nn.Embedding(options['vocab_size'], options['embed_dim'])
		self.projection = nn.Linear(options['embed_dim'], options['project_dim'])
		self.dropout = nn.Dropout(p=options['dropout_prob'])
		self.lstm = nn.LSTM(options['project_dim'], options['hidden_size'], 3)
		self.relu = nn.ReLU()
		self.out = nn.Sequential(
			nn.Linear(512, 1024),
			self.relu,
			self.dropout,
			nn.Linear(1024, 1024),
			self.relu,
			self.dropout,
			nn.Linear(1024, 1024),
			self.relu,
			self.dropout,
			nn.Linear(1024, options['out_dim'])
		)

	def forward(self, batch):
		premise_embed = self.embedding(batch.premise)
		hypothesis_embed = self.embedding(batch.hypothesis)

		premise_proj = self.relu(self.projection(premise_embed))
		hypothesis_proj = self.relu(self.projection(hypothesis_embed))

		encoded_premise, _ = self.lstm(premise_proj)
		encoded_hypothesis, _ = self.lstm(hypothesis_proj)

		premise = encoded_premise.sum(dim=1)
		hypothesis = encoded_hypothesis.sum(dim=1)
		combined = torch.cat((premise, hypothesis), 1)

		return self.out(combined)
