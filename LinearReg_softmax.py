import random
import math

def data_preprocessing(type):
	'''
	:param type: training or testing
	:return: X_train,Y_train or X_train
	'''
	import csv
	X_train = []
	Y_train = []

	classes = []
	classes_no = []
	if type == "training":filename = 'training data.txt'
	elif type == "testing":filename = 'testing data.txt'
	# with open(filename, 'r') as f:
	f = open(filename, 'r')
	reader = csv.reader(f)
	data = list(reader)
	# print(data)
	# class_count = 0
	for line in data:
		if type == "training":
			s = list(map(float, line[:-1]))
			X_train.append(s)
			if line[-1] not in classes:
				classes.append(line[-1])
				classes_no.append(classes.index(line[-1]))
				Y_train.append(classes.index(line[-1]))
			else:
				Y_train.append(classes.index(line[-1]))
		elif type == "testing":
			s = list(map(float, line[:-1]))
			X_train.append(s)
	Y_encoded = [[0 for i in range(len(classes))] for j in range(len(Y_train))]
	for i in range(len(Y_train)):
		Y = Y_train[i]
		class_index = classes_no.index(Y)
		Y_encoded[i][class_index] = 1
	# print(Y_encoded)
	# print(X_train)
	# print(Y_train)
	return X_train,Y_train,Y_encoded,classes

def intialize_parameters(dim,class_len):
	w = [[round(random.uniform(-0.001,0.001),10) for j in range(class_len)]for i in range(dim)]
	b = [0 for j in range(class_len)]
	# for i in range(dim):
	# 	for j in range(class_len)
	# 		w.append(round(random.uniform(-0.001,0.001),10))
	# 	# w.append(0)
	for i in w:
		print(i)
	parameters = {"w":w,"b":b}
	return parameters

def fwd_prop(X_train,parameters,class_len):
	w = parameters["w"]
	b = parameters["b"]
	# X_train = [[ 0.1, 0.5],[ 1.1,2.3],[-1.1,-2.3],[-1.5,-2.5]]
	# w = [[0.1, 0.2, 0.3],[0.1,0.2, 0.3]]
	# b = [0.01,0.1,0.1]
	x_rows_examples = len(X_train)
	x_cols_feature = len(X_train[0])
	z = [[0 for j in range(class_len)]for i in range(x_rows_examples)]
	for k in range(class_len):
		for i in range(x_rows_examples):
			for j in range(x_cols_feature):
				z[i][k] += (w[j][k] * X_train[i][j])
			z[i][k] += b[k]
			z[i][k] = round(z[i][k],10)
	# print("z :")
	# for val_list in z:
	# 	print(val_list)
	return z

def optimize_lin_reg_softmax(a,X_train,Y_train,Y_encoded,parameters,learning_rate):
	w = parameters["w"]
	b = parameters["b"]
	no_features = len(X_train[0])
	# calculating loss
	cost = 0
	for i in range(len(Y_train)):
		for j in range(len(a[0])):
			cost += -Y_encoded[i][j]*a[i][j]
	cost /= float(len(Y_train))
	cost = round(cost,10)
	# print("cost is ",cost)

	# gradient decent i.e calculating parameters for each class
	for class_no in range(len(a[0])):
		temp_bsum = 0
		for i in range(no_features):
			temp_wsum = 0
			for n in range(len(X_train)):
				temp_wsum += (a[n][class_no] - Y_encoded[n][class_no]) * X_train[n][i]
				temp_bsum += (a[n][class_no] - Y_encoded[n][class_no])
			w[i][class_no] = w[i][class_no] - (learning_rate * temp_wsum)
			w[i][class_no] = round(w[i][class_no], 10)
		b[class_no] = b[class_no] - (learning_rate * temp_bsum)
		b[class_no] = round(b[class_no], 10)
	parameters["b"] = b
	# print("new w: ", w)
	# print("new b:", b)

	return parameters

def softmax(z):
	z_row_len = len(z[0])
	for row in range(len(z)):
		denominator = 0
		# calculating summation of exponential of all zcols in a zrow
		for col in range(z_row_len):
			denominator += math.exp(z[row][col])
		# calculating z values for each class using softmax
		for col in range(z_row_len):
			z[row][col] = math.exp(z[row][col])/denominator
	return z

def prediction(X_test,classes,parameters):
	class_len = len(classes)
	z = fwd_prop(X_test,parameters,class_len)
	a = softmax(z)
	print 'Softmax output for X_test: ',a
	print '\n####################################################'
	for row in a:
		print 'Predicted output is: ',classes[row.index(max(row))]

def lin_reg_softmax(X_train,Y_train,Y_encoded,classes,learning_rate,iterations = 1):
	x_rows_examples = len(X_train)
	x_cols_feature = len(X_train[0])
	class_len = len(classes)
	parameters = intialize_parameters(x_cols_feature,class_len)
	w = parameters["w"]
	b = parameters["b"]
	print("x_rows_examples: ", x_rows_examples, " | x_cols_features: ", x_cols_feature)
	print(" old w:", w)
	print(" old b:", b)
	for i in range(iterations):
		z = fwd_prop(X_train, parameters,class_len)
		# print("size of z",len(z))
		a = softmax(z)
		# for row in a:
		# 	print(row)
		parameters = optimize_lin_reg_softmax(a,X_train,Y_train,Y_encoded,parameters,learning_rate)
		w = parameters["w"]
		b = parameters["b"]
	print 'new parameter w after training: ',w
	print 'new b after training: ',b
	z = fwd_prop(X_train, parameters, class_len)
	# print("size of z", len(z))
	a = softmax(z)
	print "softmax output after training is complete:"
	for row in a:
		print(row)

	# entering input for testing
	X_test = [[]]
	print "Enter test point data:"
	for i in range(len(X_train[0])):
		print 'Feature',i+1,' value: '
		feature_val = round(float(input()),2)
		X_test[0].append(feature_val)
	prediction(X_test,classes,parameters)
	return None

if __name__ == '__main__':
	X_train,Y_train,Y_encoded,classes = data_preprocessing('training')
	print("classes:",classes)
	lin_reg_softmax(X_train,Y_train,Y_encoded,classes,learning_rate=0.001,iterations=1000)
