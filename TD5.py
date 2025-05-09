#!/usr/bin/env python
# coding: utf-8

# In[1]:

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# In[ ]:





# In[4]:
# Exercice 1 :

def read_dataset(filename):
    liste = []
    with open(filename,'r')as file :
        for line in file.readlines():
            type_m, texte= line.split('\t',1)
            if type_m == 'ham':
                liste.append((0,texte))
            if type_m == 'spam' :
                liste.append((1,texte))
                
    return liste
        


# In[6]:


def spam_count(pairs):
    n = 0
    for (type_m,texte) in pairs :
        if type_m == 1 :
            n+= 1
    return n
            


# In[9]:


def transform_text(pairs):
    type_m , texte = zip(*pairs)
    vectorizer = TfidfVectorizer()
  
    X = vectorizer.fit_transform(texte)
    Y = np.array(type_m)
    
    return X, Y



# Exercice 2 :

model = KMeans(n_clusters=10, random_state=0, n_init="auto")
X, Y = transform_text(read_dataset("SMSSpamCollection"))
kmeans = model.fit(X)

# In[10]:




def kmeans_and_most_common_words(pairs, K, P):
    types, messages = zip(*pairs)
    vectorizer = TfidfVectorizer(stop_words='english')  # Ajout des stop words
    X = vectorizer.fit_transform(messages)
    order_words = vectorizer.get_feature_names_out()

    nmf_model = NMF(n_components=K, random_state=0)
    W = nmf_model.fit_transform(X)
    H = nmf_model.components_

    res = []
    for i in range(K):
        p_words = []
        top_P_indices = H[i].argsort()[-P:][::-1]
        for index in top_P_indices:
            p_words.append(order_words[index])
        res.append(p_words)

    return res

    

pairs = read_dataset('SMSSpamCollection')


K = 10
P = 5
common_words = kmeans_and_most_common_words(pairs, K, P)


for cluster_idx, words in enumerate(common_words):
    print(f"Cluster {cluster_idx + 1}: {', '.join(words)}")

#Cluster 1: ok, thanx, ask, ya, care                interaction
#Cluster 2: ll, later, sorry, meeting, text
#Cluster 3: gt, lt, min, like, wait
#Cluster 4: good, love, day, know, did
#Cluster 5: come, want, tomorrow, online, oh             message relative a des invitation
#Cluster 6: free, mobile, reply, text, ur
#Cluster 7: lor, wan, ard, dun, wat                 expression familliere
#Cluster 8: pls, send, pick, right, phone
#Cluster 9: time, wat, ur, doing, tell
#Cluster 10: home, just, reach, ready, bored

#je pense que ya une logique mais pas autant 




def best_k(pairs):

    X, Y = transform_text(pairs)
    best_score = -1
    best_k = 0
    
    for k in range(2, 10): 
        model = KMeans(n_clusters=k, random_state=0, n_init="auto")
        kmeans = model.fit(X)
        current_score = silhouette_score(X, kmeans.labels_)
        
        if current_score > best_score:
            best_score = current_score
            best_k = k
    
    return best_k, best_score
    
pairs = read_dataset("SMSSpamCollection")  # Assurez-vous que cela renvoie les bonnes paires
best_k_value, silhouette_score_value = best_k(pairs)

print(f"Meilleur K: {best_k_value}")
print(f"Score de silhouette: {silhouette_score_value:.4f}")




def classify_batch(train_pairs, test):
    
    
    # 1. Prétraitement des données
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform([text for _, text in train_pairs])
    
    # 2. Clustering avec AgglomerativeClustering
    n_clusters = 2  
    clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
    clustering_model.fit(X_train.toarray())
    
    # 3. Calculer les centroïdes 
    cluster_centers = []
    for i in range(n_clusters):
        cluster_indices = np.where(clustering_model.labels_ == i)[0]
        cluster_texts = X_train[cluster_indices].toarray()
        cluster_mean = np.mean(cluster_texts, axis=0)
        cluster_centers.append(cluster_mean)
    
    #  Prétraitement 
    X_test = vectorizer.transform(test).toarray()
    
    # Classifier les messages 
    predictions = []
    for test_vector in X_test:
        # Calculer la distance euclidienne à chaque centre de cluster
        distances = np.linalg.norm(test_vector - np.array(cluster_centers), axis=1)
        predicted_cluster = np.argmin(distances)
        predictions.append(predicted_cluster)  # 0 ou 1
    
    return predictions


