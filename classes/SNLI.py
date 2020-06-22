import torch
import torch.nn as nn
from torchtext import data, datasets
import spacy

spacy.load('en')


class SNLI():
	def __init__(self, device, batch_size=128):
		self.inputs = data.Field(lower=True, tokenize='spacy', batch_first=True)
		self.targets = data.Field(sequential=False, unk_token=None, is_target=True)
		self.train, self.dev, self.test = datasets.SNLI.splits(self.inputs, self.targets, root='./data')

		self.inputs.build_vocab(self.train, self.dev)
		self.targets.build_vocab(self.train)
		self.vocab_size = len(self.inputs.vocab)
		self.out_dim = len(self.targets.vocab)

		self.iters = data.Iterator(datasets.SNLI, batch_size=batch_size, shuffle=True, device=device)
		self.train_batches = None
		self.test_batches = None

	def generate_train_batches(self, batch_size=128):
		self.train_batches = self.iters.splits(
			datasets=(self.train,),
			batch_sizes=None,
			batch_size=batch_size
		)[0]

	def generate_test_batches(self, batch_size=128):
		self.test_batches = self.iters.splits(
			datasets=(self.test,),
			batch_sizes=None,
			batch_size=batch_size,
			shuffle=False,
			sort=False,
			sort_within_batch=False
		)[0]
