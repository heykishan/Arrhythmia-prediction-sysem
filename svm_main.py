from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from fpdf import FPDF 

import numpy as np
import matplotlib.pyplot as plt

def generateConfusionMatrixFigure(conf_arr, filename):
	font = {'size' : 17}
	plt.rc('font', **font)

	norm_conf = []
	for i in conf_arr:
	    a = 0
	    tmp_arr = []
	    a = sum(i, 0)
	    for j in i:
	        tmp_arr.append(float(j)/float(a))
	    norm_conf.append(tmp_arr)

	fig = plt.figure()
	plt.clf()
	ax = fig.add_subplot(111)
	ax.set_aspect(1)
	res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
	                interpolation='nearest')

	width = len(conf_arr)
	height = len(conf_arr[0])

	for x in xrange(width):
	    for y in xrange(height):
	        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
	                    horizontalalignment='center',
	                    verticalalignment='center')

	cb = fig.colorbar(res)
	axis_labels = ['1','2','3','4','5','6','7','8','9','10','14','15','16']
	plt.xticks(range(width), axis_labels)
	plt.yticks(range(height), axis_labels)

	plt.savefig(filename + '.eps', format='eps')
	plt.savefig(filename + '.png', format='png')

input_file = open("data/data_clean_imputed.csv","r")
# input_file = open("../data/pca.csv","r")

lines = input_file.readlines()

TRAINING_SIZE = 316

print("\nEnter Classification type required - \n1. Binary 2. Multiclass")
class_type = int(input())
CLASSIFICATION_TYPE = class_type
NUM_PCA = 270;

X = []
y = []

test_X = []
test_y = []

count = 0
for line in lines:
	tokens = line.strip().split(",")
	if count < TRAINING_SIZE:
		X.append(map(float, tokens[0:NUM_PCA]))
		if CLASSIFICATION_TYPE == 2:
			y.append(int(tokens[len(tokens)-1]))
		elif int(tokens[len(tokens)-1]) == 1:
			y.append(0)
		else:
			y.append(1)
		count += 1
	else:
		test_X.append(map(float, tokens[0:NUM_PCA]))
		if CLASSIFICATION_TYPE == 2:
			test_y.append(int(tokens[len(tokens)-1]))
		elif int(tokens[len(tokens)-1]) == 1:
			test_y.append(0)
		else:
			test_y.append(1)


train_predictions = OneVsRestClassifier(LinearSVC()).fit(X, y).predict(X)
test_predictions = OneVsRestClassifier(LinearSVC()).fit(X, y).predict(test_X)

test_file = open("../data/ecg_trial.txt","r")
data_corr = 1

train_missed = 0
test_missed = 0

for i in range(0, len(train_predictions)):
	if train_predictions[i] != y[i]:
		train_missed += 1
for i in range(0, len(test_predictions)):
	if test_predictions[i] != test_y[i]:
		test_missed += 1

train_error = train_missed*1.0/len(train_predictions)
test_error = test_missed*1.0/len(test_predictions)

print "TRAIN RESULTS"
print "Train Accuracy: " + str(1-train_error)
print "Confusion Matrix (Train)"
conf_array_train = confusion_matrix(y, train_predictions)
generateConfusionMatrixFigure(conf_array_train, 'svm_train')
print conf_array_train

print "TEST RESULTS"
print "Test Accuracy: " + str(1-test_error)
print "Confusion Matrix (Test)"
conf_arr_test = confusion_matrix(test_y, test_predictions)
generateConfusionMatrixFigure(conf_arr_test, 'svm_test')


print conf_arr_test
print type(conf_arr_test)
print"Classification Report (Test)"
print classification_report(test_y, test_predictions)



pdf = FPDF()
pdf.add_page()
pdf.set_font('Times', 'B', 16)
pdf.cell(50, 10, "TRAIN RESULTS", 0, 1)
pdf.set_font('Times', 'B', 10)
pdf.cell(50, 10, "Train Accuracy: " + str(1-train_error), 0, 1)
pdf.cell(50, 10, "Test Accuracy: " + str(1-test_error), 0, 1)

if(data_corr):
	pdf.cell(50, 10, "Test Result: Normal. Healthy data.", 0, 1)
else:
	pdf.cell(50, 10, "Test Result: Abnormal. Consulting of a doctor is recommended.", 0, 1)

pdf.cell(50, 10, "Accuracy of prediction: " + str(round((1-train_error),3)) , 0, 1)

# pdf.image("C:\Users\Kishan B\Desktop\Code\svm_test.png")
# pdf.output('generated_report.pdf', 'F')