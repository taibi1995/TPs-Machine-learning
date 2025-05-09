import math
import random
import numpy as np
from sklearn.model_selection import KFold



# Exercice 1: Lecture du dataset, encodage


def read_data(filename):
   
    
    X, Y = [], []
   
    with open(filename, 'r') as file:
        
        for line in file:

            values = line.strip().split(',')
            X.append([float(value) for value in values[2:]])
            Y.append(values[1] == 'M')
    
    return X, Y
    
    
    
    
# Exercice 2: Distance euclidienne


def simple_distance(data1, data2):
    
    x = sum((a - b) ** 2 for a, b in zip(data1, data2))
    
    return math.sqrt(x)




#Exercice 3: K Nearest Neighbors


def k_nearest_neighbors(x, points, dist_function, k):
   

    distances = [(i, dist_function(x, point)) for i, point in enumerate(points)]
    distances.sort(key=lambda x: x[1])
    neighbors = [i for i, _ in distances[:k]]
    
    return neighbors




#Exercice 4: Prédiction

def split_lines(input, seed, output1, output2):
 
  random.seed(seed)
  out1 = open(output1, 'w')
  out2 = open(output2, 'w')
  for line in open(input, 'r').readlines():
    if random.randint(0, 1):
      out1.write(line)
    else:
      out2.write(line)


def is_cancerous_knn(x, train_x, train_y, dist_function, k):
   
    
    indices_voisins_proches = k_nearest_neighbors(x, train_x, dist_function, k)
    etiquettes_voisins = [train_y[i] for i in indices_voisins_proches]
    
    return 2 * sum(etiquettes_voisins) >= k



# Exercice 5: Évaluation

def eval_cancer_classifier(test_x, test_y, classifier):
   
    erreurs = 0  
    n = len(test_x)  
    
    for x, y_true in zip(test_x, test_y):
        y_pred = classifier(x)  
        if y_pred != y_true:
            erreurs += 1  

    taux_erreur = erreurs / n
    
    return taux_erreur


# Exercice 6: Validation Croisée 1 / 2: Évaluation sur l'ensemble d'entraînement


def cross_validation(train_x, train_y, untrained_classifier):
   
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  
    
    error_rates = []
    
    for train_indices, test_indices in kf.split(train_x):

        block_train_x = [train_x[i] for i in train_indices]
        block_train_y = [train_y[i] for i in train_indices]
        block_test_x = [train_x[i] for i in test_indices]
        block_test_y = [train_y[i] for i in test_indices]
        

        error_rate = 0
        for i in range(len(block_test_x)):

            prediction = untrained_classifier(block_train_x, block_train_y, block_test_x[i])
            error_rate += (prediction != block_test_y[i])
        

        error_rates.append(error_rate / len(block_test_x))
    

    return sum(error_rates) / 5


# Exercice 7: Validation Croisée 2 / 2: Optimisation de paramètre



def sampled_range(mini, maxi, num):
  if not num:
    return []
  lmini = math.log(mini)
  lmaxi = math.log(maxi)
  ldelta = (lmaxi - lmini) / (num - 1)
  out = [x for x in set([int(math.exp(lmini + i * ldelta)) for i in range(num)])]
  out.sort()
  return out



def find_best_k(train_x, train_y, untrained_classifier_for_k):
   
    
    l = sampled_range(2, len(train_x), 20)
    error_rate = 1
    best_k = None

    for k in l:
        error_k = cross_validation(train_x, train_y, lambda X, Y, x: untrained_classifier_for_k(X, Y, k, x))
        if error_k < error_rate:
            best_k = k
            error_rate = error_k

    return best_k



# Exercice 8 (*): Distance pondérée.



def get_weighted_dist_function(train_x, train_y):
    
    train = np.array(train_x)
    ecart_type = np.std(train, axis=0)  


    return lambda x1, x2: math.sqrt( sum(((x1[i] - x2[i]) ** 2) / ecart_type[i] for i in range(len(x1))))




split_lines("wdbc.data", 1, "train", "test" )
train_X, train_Y = read_data("train")
test_X, test_Y = read_data("test")


untrained_classifier_for_k_weighted = lambda train_x, train_y, k, x: is_cancerous_knn(x, train_x, train_y, get_weighted_dist_function(train_x, train_y), k)


k_weighted = find_best_k(train_X, train_Y, untrained_classifier_for_k_weighted)
print(f"Meilleur K pour distance pondérée : {k_weighted}")


classifier_weighted = lambda x: is_cancerous_knn(x, train_X, train_Y, get_weighted_dist_function(train_X, train_Y), k_weighted)
performance_weighted = eval_cancer_classifier(test_X, test_Y, classifier_weighted)
print(f"Performance avec distance pondérée pour K={k_weighted} : {performance_weighted}")


classifier_simple = lambda x: is_cancerous_knn(x, train_X, train_Y, simple_distance, 12)  # Utiliser K=12 pour simple distance
performance_simple = eval_cancer_classifier(test_X, test_Y, classifier_simple)
print(f"Performance avec distance simple pour K=12 : {performance_simple}")


for k in [3, 5, 7, 10, 15, 20, 25]:  

    classifier_simple_k = lambda x: is_cancerous_knn(x, train_X, train_Y, simple_distance, k)
    performance_simple_k = eval_cancer_classifier(test_X, test_Y, classifier_simple_k)
    print(f"Performance avec distance simple pour K={k} : {performance_simple_k}")
    

    classifier_weighted_k = lambda x: is_cancerous_knn(x, train_X, train_Y, get_weighted_dist_function(train_X, train_Y), k)
    performance_weighted_k = eval_cancer_classifier(test_X, test_Y, classifier_weighted_k)
    print(f"Performance avec distance pondérée pour K={k} : {performance_weighted_k}")


#Meilleur K pour distance pondérée : 3
#Performance avec distance pondérée pour K=3 : 0.03914590747330961
#Performance avec distance simple pour K=12 : 0.060498220640569395
#Performance avec distance simple pour K=3 : 0.05693950177935943
#Performance avec distance pondérée pour K=3 : 0.03914590747330961
#Performance avec distance simple pour K=5 : 0.05693950177935943
#Performance avec distance pondérée pour K=5 : 0.03558718861209965
#Performance avec distance simple pour K=7 : 0.060498220640569395
#Performance avec distance pondérée pour K=7 : 0.060498220640569395
#Performance avec distance simple pour K=10 : 0.05693950177935943
#Performance avec distance pondérée pour K=10 : 0.03914590747330961
#Performance avec distance simple pour K=15 : 0.06761565836298933
#Performance avec distance pondérée pour K=15 : 0.06761565836298933
#Performance avec distance simple pour K=20 : 0.06761565836298933
#Performance avec distance pondérée pour K=20 : 0.060498220640569395
#Performance avec distance simple pour K=25 : 0.0711743772241993
#Performance avec distance pondérée pour K=25 : 0.06405693950177936 

# remarque :
#avec distance pondérée on a une meilleurs performance  et donne de meilleurs resultat mais pour un k plus grand que 14  la distance simple donne de un meilleurs resultat legement 
