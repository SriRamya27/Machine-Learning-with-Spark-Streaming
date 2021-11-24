import pickle
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans

nb_model = BernoulliNB()
# save the model to disk
filename = 'nb_finalized_model.sav'
with open(filename, 'wb') as f:
    pickle.dump(nb_model, f)
 
    
perceptron_model =  Perceptron(tol=1e-3, random_state=0)
# save the model to disk
filename = 'perceptron_finalized_model.sav'
with open(filename, 'wb') as f:
    pickle.dump(perceptron_model, f)
  
    
sgd_model = SGDClassifier(max_iter=1000, tol=0.01)
# save the model to disk
filename = 'sgd_finalized_model.sav'
with open(filename, 'wb') as f:
    pickle.dump(sgd_model, f)


kmeans_cluster = KMeans(n_clusters=2)
# save the cluster to disk
filename = 'kmeans_finalized_cluster.sav'
with open(filename, 'wb') as f:
    pickle.dump(kmeans_cluster, f)


