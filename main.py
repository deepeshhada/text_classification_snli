import utilities.deep_util as deep_util
import utilities.general_util as general_util

from classes.SNLI import SNLI
from classes.InferNet import InferNet

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
import pickle


def deep_model():
	device = deep_util.get_device()
	snli = SNLI(device=device, batch_size=128)
	snli.generate_test_batches(batch_size=128)
	target_dict = deep_util.get_target_dict(snli.targets)

	torch.manual_seed(0)
	model = deep_util.load_trained_model()

	preds = []
	for i, batch in enumerate(snli.test_batches):
		outputs = model(batch).to(device)
		batch_preds = torch.max(outputs, 1)[1].view(batch.label.size()).tolist()
		preds.extend([target_dict[pred] for pred in batch_preds])

	general_util.print_file(output_list=preds, file_name="deep_model.txt")
	return


def tfidf_model():
	X_train, Y_train = general_util.generate_data(path='./data/snli/snli_1.0/snli_1.0_train.jsonl')
	X_test, Y_test = general_util.generate_data(path='./data/snli/snli_1.0/snli_1.0_test.jsonl')
	vectorizer = TfidfVectorizer()
	X_train_transformed = vectorizer.fit_transform(X_train)
	vocab = vectorizer.vocabulary_
	vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, vocabulary=vocab, analyzer="word", tokenizer=word_tokenize)

	model = pickle.load(open("./models/tfidf_model", 'rb'))
	X_test_transformed = vectorizer.fit_transform(X_test)
	preds = model.predict(X_test_transformed)
	general_util.print_file(output_list=preds, file_name="tfidf.txt")
	return


general_util.extract_data()
deep_model()
tfidf_model()

print("Name = Deepesh Virendra Hada")
print("IISc SR No. = 17196")
print("Course = M.Tech.")
print("Department = CSA")
