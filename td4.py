#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
     
# Exercice 01 :

def split_lines(input, seed, output1, output2):
    
    random.seed(seed)
    with open(input,'r') as canalin :
        with open(output1,'w') as canalout1 :
            with open(output2,'w') as canalout2:
            
                for line in canalin.readlines():
                    if (random.random() <= 0.5):
                        canalout1.write(line)
                    else :
                        canalout2.write(line)
                        


# In[3]:

     
# Exercice 02 :

def tokenize_and_split(sms_file):
    words=dict()
    l0,l1 = [],[]
    with open(sms_file,'r') as file :
        for line in file.readlines():
            type_phrase , phrase =line.split(None,1)
            tokens = phrase.split()
            indices=[] 
            for word in tokens :
                if not (word in words):
                    words[word]=len(words)
                indices.append(words[word])
            
            if type_phrase =='spam' :
                l0.append(indices)
            if type_phrase == 'ham':
                l1.append(indices) 
    return words , l0 ,l1
                

  


# In[4]:
     
# Exercice 03 :

def compute_frequencies(num_words, documents):
    
    liste = [0]*num_words
    for document in documents :
        for word in set(document):
            liste[word]+=1
    return [x /len(documents) for x in liste]


# In[5]:

     
# Exercice 04 :


def naive_bayes_train(sms_file): 

    words , l0, l1 = tokenize_and_split(sms_file)
    r_spam = len(l0)/ (len(l0) + len(l1))
    taille = len(words)
    freq_spam = compute_frequencies(taille,l0)
    freq_total=  compute_frequencies(taille,l0+l1)
    spamicity = []
    for i in range(taille):
        spamicity.append(freq_spam[i]/freq_total[i])
    return r_spam , words , spamicity 
        


# In[6]:

     
# Exercice 05 :

def naive_bayes_predict(train_spam_ratio, train_words, train_spamicity, sms):
 
    
    sms_words = sms.split()
    
    p = train_spam_ratio
    
    for word in set(sms_words) :
        if word in train_words :
            p *= train_spamicity[train_words[word]]
    
      
    return p
    
     
# Exercice 06 :


def naive_bayes_eval(test_sms_file, f):
    nb_spam = 0
    nb_spam_f = 0
    nb_spam_no_erreur = 0
    
    precision = 1.0
    
    with open(test_sms_file, "r") as file:
        for line in file.readlines():
            msg_type, content = line.split(None, 1)
            pred_spam = f(content)
            
            if msg_type == "spam":
                nb_spam += 1
                if pred_spam:
                    nb_spam_no_erreur += 1
            
            nb_spam_f += pred_spam
    
    if nb_spam_f != 0:
        precision = nb_spam_no_erreur / nb_spam_f

    recall = nb_spam_no_erreur / nb_spam if nb_spam > 0 else 0.0
    
    return recall, precision
   
# In[ ]:
# Exercice 07 :


split_lines("SMSSpamCollection", 1, "train", "test")
r_spam, words, spamicity = naive_bayes_train("train")

print( naive_bayes_eval("test",lambda x:naive_bayes_predict(r_spam, words, spamicity, x)>0.6))


def classify_spam(sms):

    return naive_bayes_predict(r_spam, words, spamicity, sms) > 0.9




# Exercice 08 :

def naive_bayes_train_ham(sms_file):
   
    
    words, l0, l1 = tokenize_and_split(sms_file)
    
    r_ham = len(l1)  / (len (l0) + len(l1))
    freq_ham = compute_frequencies(len(words), l1)
    freq_total = compute_frequencies(len(words), l1+l0)
    
    hamicity = []
    
    for i in range(len(words)) :
            hamicity.append(freq_ham[i]/freq_total[i])
            
    return r_ham, words, hamicity

    


r_ham, words, hamicity = naive_bayes_train_ham("train")

print( naive_bayes_eval("test",lambda x:naive_bayes_predict(r_ham, words, hamicity, x)<=0.1))




def classify_spam_precision(sms):
    
    
    return naive_bayes_predict(r_spam, words, spamicity, sms) > 0.7



def classify_spam_recall(sms):
   
   
   
    return naive_bayes_predict(r_ham, words, hamicity, sms) <= 0.1








