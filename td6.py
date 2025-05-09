from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

# In[3]:


# Exercice 1 :

data=open('jester_jokes.txt','r').readlines()
vectorizer = TfidfVectorizer()    
tfidf = vectorizer.fit_transform(data)



def most_similar(tfidf, item_index, k):
  
    

    similarities = cosine_similarity(tfidf)
    similarity_item = similarities[item_index]
    indices_sorted = sorted(range(len(similarity_item)), key=lambda i: similarity_item[i], reverse=True)
    similar_items = [i for i in indices_sorted if i != item_index][:k]
    
    return similar_items




# Exercice 2:


def read_ratings(filename, num_jokes):

    
    res = defaultdict(dict)
    

    with open(filename, "r") as canalIn:
        for line in canalIn:
            line = line.strip().split(",")
            user_id = int(line[0])
            joke_id = int(line[1])
            rating = float(line[2])
            res[user_id][joke_id] = rating
    
    return list(res.values())




def content_recommend(similarity_matrix, user_ratings, k):
   
    

    notes_blagues_paires = {blague_id: note for blague_id, note in user_ratings.items() if blague_id % 2 == 0}
    

    ratings_blagues_impaires = []
    

    for blague_impaire_id in range(1, similarity_matrix.shape[0], 2):  
        somme_ponderee = 0
        somme_similarite = 0
        
        for blague_paire_id, note in notes_blagues_paires.items():
            somme_ponderee += note * similarity_matrix[blague_impaire_id, blague_paire_id]
            somme_similarite += similarity_matrix[blague_impaire_id, blague_paire_id]
        

        score = somme_ponderee / somme_similarite if somme_similarite != 0 else 0
        ratings_blagues_impaires.append((blague_impaire_id, score))
    

    ratings_blagues_impaires.sort(key=lambda x: x[1], reverse=True)
    top_ratings = [blague_id for blague_id, _ in ratings_blagues_impaires[:k]]
    
    return top_ratings




# Exercice 3 :

def collaborative_recommend(ratings, user_ratings, k):
    
   
    recommendations = [] 
    

    rated_jokes = user_ratings.keys()
    unseen_jokes = [joke for joke in range(150) if joke not in rated_jokes]
    

    for joke in unseen_jokes:
        numerator = 0
        denominator = 0
        
        # Comparer avec les évaluations des autres utilisateurs
        for other_user in ratings:
            if joke in other_user:
                common_ratings = []
                user_ratings_for_common = []
                
                # Trouver les blagues communes entre l'utilisateur et l'autre
                common_keys = rated_jokes & other_user.keys()
                
                # Rassembler les évaluations communes
                for key in common_keys:
                    common_ratings.append(other_user[key])
                    user_ratings_for_common.append(user_ratings[key])
                
                # Calculer la corrélation entre les évaluations des deux utilisateurs
                correlation_matrix = np.corrcoef(np.array([common_ratings, user_ratings_for_common]))
                
                numerator += correlation_matrix[0, 1] * other_user[joke]
                denominator += correlation_matrix[0, 1]
        
        # Calculer le rating estimé si possible
        if denominator != 0:
            recommendations.append((joke, numerator / denominator))
    
    # Trier les résultats par score décroissant et renvoyer les k meilleurs
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return [joke for joke, _ in recommendations[:k]]


