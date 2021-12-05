'''
python3 stream.py -f sentiment -b 10000 
$SPARK_HOME/bin/spark-submit stream_preprocess.py > output.txt
'''

from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import numpy as np
import pickle
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans, KMeans
import matplotlib.pyplot as plt

def testNaiveBayes(x,y):	
	# load the model from disk
	nb_loaded_model = pickle.load(open('nb_finalized_model.sav', 'rb'))
	print("in nb")
	xnew= nb_loaded_model.predict(x)
	print_evaluate(y, xnew)
	
#naiveBayes()	
	
def testPerceptron(x,y):	
	# load the model from disk
	perceptron_loaded_model = pickle.load(open('perceptron_finalized_model.sav', 'rb'))
	print("in preceptron")
	xnew= perceptron_loaded_model.predict(x)
	print_evaluate(y, xnew)
#perceptron()


def testSdg(x,y):	
	# load the model from disk
	sgd_loaded_model = pickle.load(open('sgd_finalized_model.sav', 'rb'))
	print("in sdg")
	xnew= sgd_loaded_model.predict(x)
	print_evaluate(y, xnew)
#sgd()	


def testKmeans(x,y):	
	# load the cluster from disk
	kmeans_loaded_cluster = pickle.load(open('kmeans_finalized_cluster.sav', 'rb'))
	print("in kmeans")
	xnew= kmeans_loaded_cluster.predict(x)
	print_evaluate(y, xnew)
#kmeans()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print(confusion_matrix(true, predicted))
    print(classification_report(true, predicted))
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('______________________')
    lineGraph(true,predicted)

def lineGraph(x, y):
    plt.plot(x, y, linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='blue', markersize=12)
    plt.ylim(0,1.5)
    plt.xlim(0,1.5)
    
    plt.show()


