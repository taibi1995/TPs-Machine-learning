```python
import math
from random import random
```

# Exercice 1 – Listes, dictionnaires, ensembles



def averge(lst):
    
    
    return sum(lst)/len(lst)


def averge (lst):
    
    somme = 0
    size = 0
    
    for i in lst :
        somme += i
        size +=1
    
    return somme/size 



    """
averge([3, 7, 1, 2, 3])





    3.2"""



# fonction median :


```python
def mediane (lst):
    lst.sort()
    l = len(lst)
    return lst[(l//2)]
```


```python
l= [2, 4, 5, 6, 173]
mediane (l)

```




    5




"""
averge(l)





    38.0
"""


# bad_sample :


```python
def bad_sample(size ,delta):
    l = [0]* size 
    l[-1]= delta * size
    return l
```


"""
bad_sample(4,3)
```




    [0, 0, 0, 12]"""





def bad_sample2(size,avg,med):
    l = [med]*size
    delta = avg- med
    if delta < 0 :
        l[0] += delta * size
    
    else :
        l[-1] += delta * size
    
    return l 
```

# nombre d'occurences :



def occurrences(lst):
    d = {}
    for i in lst :
        if i in d.keys() :
            
            d[i] += 1
        else:
            d[i]= 1
    return d
            
    
```


"""
l= [1, 2, 4, 3, 4, 1, 1, 3]
occurrences(l)





    {1: 3, 2: 1, 4: 2, 3: 2}"""



# Unique :



def unique(lst):
    l =[]
    for i in lst :
        if i not in l :
            l.append(i)
    return l
```


"""
l= [2, 3, 1, 1, 3, 1, 4, 1, 3]
unique(l)
```




    [2, 3, 1, 4]"""



# Exercice 2 :
# l’écart-type d’une liste :




def squares(lst):
    
    return [x**2 for x in lst]
```


"""
l=[1, 2, 3, 4]
squares(l)
```




    [1, 4, 9, 16]"""





def stddv(lst):
    avg= averge(lst)
    s =  [(x-avg)**2 for x in lst]
    return math.sqrt(averge(s))
        
```


"""
l = [5, 4, 7, 4, 4, 2, 5, 9]
stddv(l)

```




    2.0"""




"""
l = [20, 20, 20]
stddv(l)
```




    0.0
"""




def quicksort_(lst):
    
    size = len(lst)
    
    if size <= 1 :
        return lst
    
    pivot = lst[0]
    
    gauche = [x for x in lst if x < pivot]
    milieu = [x for x in lst if x == pivot]
    droite = [x for x in lst if x > pivot]
    
    return quicksort(gauche) + milieu + quicksort(droite)
    
```

# Exercice 3 – Simulation de variables aléatoires :

# Uniforme :


```python
def uniforme():
    x = random()
    if x < 0 :
        return 0
    return 1
```



uniforme()
"""




    1"""





def uniforme():
    x = random()*6
    return math.floor(x)
    
```



def uniforme(n):
    x = random()*n
    return math.floor(x)
```



uniforme(7)
"""




    5


"""
# Exam succes :



def exam_succes(n,p):
    exam_reussi= 0
    for i in range(n):
        if random()<p:
            exam_reussi+=1
    return exam_reussi
```


"""
exam_succes(5,0.5)
```




    4
"""


# Exercice 4 – Paradoxe de Monty Hall:



def monty_hall(change):
   
    portes = [0, 1, 2]
    
   
    portevoiture = uniforme(3)
    
    portecandidat = uniforme(3)
    
    portesdispo= [p for p in portes if p != portevoiture and
                                          p != portecandidat
                        ]
    
    portePres = portesdispo[uniforme(len(portesdispo))]
   
    
    if change :
        
        portecandidat = [p for p in portes if p != portepres  and
                                              p != portecandidat ] [0]
   
    return portecandidat == portevoiture

```



def monty_hall_simulation(n):
    
    
    nb_true = 0
    nb_false = 0
    
    for i in range(n) :
        nb_true += monty_hall(True)        
        nb_false += monty_hall(False)
        
    return (nb_true/n, nb_false/n)

```


"""
monty_hall_simulation(1000000)
```




    (0.666833, 0.332725)"""

"""

# conclusion :

# changer de porte est la strategie gagnate dans le jeu d monty hall """




