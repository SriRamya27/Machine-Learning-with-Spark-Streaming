import pickle
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans, KMeans

nb_model = BernoulliNB(alpha=0.1)
# save the model to disk
filename = 'nb_finalized_model.sav'
with open(filename, 'wb') as f:
    pickle.dump(nb_model, f)
    
perceptron_model =  Perceptron(alpha=0.01, tol=1e-3, random_state=0)
# save the model to disk
filename = 'perceptron_finalized_model.sav'
with open(filename, 'wb') as f:
    pickle.dump(perceptron_model, f)
  
    
sgd_model = SGDClassifier(alpha=0.01, max_iter=1000, tol=0.01, learning_rate='optimal')
# save the model to disk
filename = 'sgd_finalized_model.sav'
with open(filename, 'wb') as f:
    pickle.dump(sgd_model, f)



kmeans = MiniBatchKMeans(n_clusters=2, max_iter=1000, tol=1.0, max_no_improvement=20, reassignment_ratio=0.0001)
# save the model to disk
filename = 'kmeans_finalized_cluster.sav'
with open(filename, 'wb') as f:
    pickle.dump(kmeans, f)


