#!/usr/bin/env python
# coding: utf-8

# In[1]:


from random import random
import math
import numpy as np


# # Exercice 0:

# In[14]:


def simulation_coin(num_exp, num_coins_per_exp, num_buckets):
    buckets= [0] * (num_buckets+1)
    for i in range(num_exp):
        num_tails=0
        for j in range (num_coins_per_exp):
            if (random()<=0.5):
                num_tails+=1
        r = num_tails/num_coins_per_exp   
        i = int(r * num_buckets)
        buckets[i]+=1
    return buckets   
    


# # Exercice 1:

# In[47]:


def proba_normal_var_above(value):
    
    return (math.erfc(value / math.sqrt(2))) / 2


# # Exercice 2:

# In[48]:


def proba_sample_mean_above_true_mean_by_at_least(sample, delta):
    n = len(sample)
    moyenne_sample = sum(sample)/n
    moyenne = moyenne_sample - delta
    variance = sum([(x-moyenne_sample)**2 for x in sample])/(n-1)
    if delta == 0 :
        return proba_normal_var_above(0)
    if variance == 0 :
        return moyenne_sample <= moyenne
    
    value = delta * math.sqrt(n)/ math.sqrt(variance)
    return proba_normal_var_above(value)
    
    


# # Exercice 3:
# 

# In[49]:


def standard_percentile(p):
    if p < 0 or p > 1:
        raise ValueError("p doit être un nombre entre 0 et 1")
    sup= 50
    inf = -50
    while sup - inf > 1e-6 :
        m = (sup + inf)/2
        prob= 1 - proba_normal_var_above(m)
        if prob <  p :
            inf = m
        else :
            sup = m
    return (sup+inf)/2


# # Exercice 4:

# In[50]:


def confidence_interval_of_mean(sample, pvalue):

    n = len(sample)
    moyenne_sample = sum(sample)/n
    variance = sum([(x - moyenne_sample) ** 2 for x in sample]) / (n - 1)
    
    ecart_type= math.sqrt(variance)
    x= pvalue / 2
    value_min = -standard_percentile(x)
    value_max = standard_percentile(1-x)
    borne_min = moyenne_sample - value_min * (ecart_type / math.sqrt(n))
    borne_max = moyenne_sample + value_max * (ecart_type / math.sqrt(n))
    return (borne_min,borne_max)


# In[ ]:




