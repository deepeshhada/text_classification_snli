import torch
import torchtext
import json
import jsonlines
import zipfile
import os


def extract_data():
	src_path = "./data/snli_1.0.zip"
	dest_path = "./data"

	with zipfile.ZipFile(src_path, 'r') as f:
		f.extractall(dest_path)
	return


def generate_data(path):
	X = []
	Y = []

	with jsonlines.open(path) as f:
		for line in f.iter():
			json_str = json.dumps(line)
			example = json.loads(json_str)
			if example['gold_label'] != '-':
				X.append(example['sentence1'] + ' <bawaalend> ' + example['sentence2'])
				Y.append(example['gold_label'])

	return X, Y


def print_file(output_list, file_name):
	with open(file_name, 'w') as f:
		for index in range(len(output_list)):
			if index != len(output_list) - 1:
				f.write("%s\n" % (output_list[index]))
			else:
				f.write("%s" % (output_list[index]))
	f.close()
	return


