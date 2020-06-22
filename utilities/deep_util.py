import torch
import torchtext
from classes.InferNet import InferNet


def evaluation(batches, model, device):
	total, correct = 0, 0
	for i, batch in enumerate(batches):
		outputs = model(batch).to(device)
		correct += (torch.max(outputs, 1)[1].view(batch.label.size()) == batch.label).sum().item()
		total += batch.batch_size

	return 100 * correct / total


def generate_options():
	return {
		'embed_dim': 300,
		'project_dim': 300,
		'hidden_size': 256,
		'dropout_prob': 0.2,
		'out_dim': 3,
		'vocab_size': 33932
	}


def get_device():
	return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_target_dict(targets):
	target_dict = dict(targets.vocab.stoi)
	target_dict = dict((v, k) for k, v in target_dict.items())
	return target_dict


def load_trained_model(device=get_device()):
	options = generate_options()
	model = InferNet(options).to(device)
	model.load_state_dict(torch.load("./models/deep_model", map_location=torch.device('cpu')))
	return model
