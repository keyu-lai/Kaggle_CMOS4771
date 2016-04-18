import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
import numpy as np

def get_data(filename, numeric_fields, idx_to_field):

	"""
	This function will convert all the data from the input file into a list of dictionaries
	Output:
		y will be None if it's a test set
	"""
	
	# Read all the data from the input file
	data = list()
	with open(filename, 'r') as csvfile:
		csvreader = csv.reader(csvfile)
		for row in csvreader:
			data.append(row)

	# Distinct from tarining set and test set
	# Also get the labels for the training set
	y = list()
	if data[0][-1] == 'label':
		l = len(data[0]) - 1
		for row in data[1:]:
			y.append(int(row[-1]))
	else:
		l = len(data[0])
		y = None

	# convert to a dictionary, which could be further fed into DictVectorizer
	X = list()
	for row in data[1:]:
		tmp = dict()
		for i in range(l):
			field = idx_to_field[i]
			if field in numeric_fields:
				tmp[field] = float(row[i])
			else:
				tmp[field] = row[i]
		X.append(tmp)

	return X, y

def get_numeric(filename):

	"""
	Output:
		numeric_fields: all the numeric fields
		idx_to_field: convert a index to corresponding field name
	"""

	numeric_fields = set()
	idx_to_field = list()
	with open(filename, 'r') as file:
		for row in file.readlines():
			row = row.strip().split(' ')
			if row[1] == 'numeric':
				numeric_fields.add(row[0])
			idx_to_field.append(row[0])
	return numeric_fields, idx_to_field

if __name__ == '__main__':

	numeric_fields, idx_to_field = get_numeric('field_types.txt')
	training_set, training_labels = get_data('data.csv', numeric_fields, idx_to_field)
	# test_set, _ = get_data('quiz.csv', numeric_fields, idx_to_field)

	sep = int(len(training_set) * 0.9)
	test_set = training_set[sep:]
	test_labels = training_labels[sep:]
	training_set = training_set[:sep]
	training_labels = training_labels[:sep]
	v = DictVectorizer()

	training_X = v.fit_transform(training_set).toarray()
	training_y = np.array(training_labels)
	test_X = v.transform(test_set).toarray()
	test_y = test_labels

	classifer = SVC()
	classifer.fit(training_X, training_y)
	res = classifer.predict(test_X).tolist()
	count = 0
	for i in range(len(res)):
		if res[i] == test_y[i]:
			count += 1
	print float(count) / len(res)


