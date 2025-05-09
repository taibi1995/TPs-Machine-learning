import math
import random
import numpy as np
from sklearn.svm import SVC
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

# Exercice 8 : SVM


def svm_classify(train_x, train_y, X):
   

    model = SVC()
    model.fit(train_x,train_y)
    
    return model.predict(X)


def svm_classify_with_param(train_x, train_y, c, X):
    
    model = SVC(C=c, kernel='rbf')
    model.fit(train_x, train_y)
    return model.predict(X).tolist()


def find_best_c(train_x, train_y, untrained_classifier_for_c):
    c_values = [0.01, 0.1, 1, 10, 100, 1000,10000,100000,200000]
    best_c = None
    best_error = float('inf')
    
    for c in c_values:

        untrained_classifier = lambda tx, ty: lambda x: untrained_classifier_for_c(tx, ty, c, x)
        error_rate = cross_validation_with_batch(train_x, train_y, untrained_classifier)
        
        if error_rate < best_error:
            best_c = c
            best_error = error_rate
    
    return best_c




def cross_validation_with_batch(train_x, train_y, untrained_classifier):
    kf = KFold(n_splits=5)
    error_rates = []
    
    for train_indices, test_indices in kf.split(train_x):
        block_train_x = [train_x[i] for i in train_indices]
        block_train_y = [train_y[i] for i in train_indices]
        block_test_x = [train_x[i] for i in test_indices]
        block_test_y = [train_y[i] for i in test_indices]
        
        trained_classifier = untrained_classifier(block_train_x, block_train_y)
        predictions = trained_classifier(block_test_x)
        error_rate = sum(p != t for p, t in zip(predictions, block_test_y)) / len(block_test_y)
        error_rates.append(error_rate)
    
    return sum(error_rates) / len(error_rates)



# Exercice 9: SVM avec similarité injectée


def svm_classify_dist(train_x, train_y, distance_function, X):
   
    similarity_matrix = np.array([
        [1 / (1 + distance_function(x_train, x_test)) for x_test in X] 
        for x_train in train_x ])
    

    model = SVC(kernel='linear')
    model.fit(similarity_matrix, train_y)
    return model.predict(similarity_matrix)


split_lines("wdbc.data", 1, "train", "test" )
train_X, train_Y = read_data("train")
test_X, test_Y = read_data("test")
classifier = lambda tx, ty, c, x: svm_classify_with_param(tx, ty, c, x)
c = find_best_c(train_X, train_Y, classifier)
print(c)
classifier2 = lambda x_: svm_classify_with_param(train_X, train_Y, c, [x_])[0]
print(eval_cancer_classifier(test_X, test_Y, classifier2))


#100000
#0.028469750889679714  taux d erreur parfait / au td7 avec la distance ponderé pour k= 3 jai eu 0.03914590747330961

