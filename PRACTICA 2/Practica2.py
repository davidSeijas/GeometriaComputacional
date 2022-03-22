"""
DAVID SEIJAS PÉREZ
PRÁCTICA 2
"""

import math
import pandas as pd
  

with open('GCOM2022_pract2_auxiliar_eng.txt', 'r', encoding="utf8") as file:
      en = file.read()
     
with open('GCOM2022_pract2_auxiliar_esp.txt', 'r', encoding="utf8") as file:
      es = file.read()
      
from collections import Counter
tab_en = Counter(en)
tab_es = Counter(es)

tab_en_states = list(tab_en)
tab_en_weights = list(tab_en.values())
tab_en_probab = [x/float(sum(tab_en_weights)) for x in tab_en_weights]
distr_en = pd.DataFrame({'states': tab_en_states, 'probab': tab_en_probab})
distr_en = distr_en.sort_values(by='probab', ascending=True)
distr_en.index = [i for i in range(len(tab_en_states))]

tab_es_states = list(tab_es)
tab_es_weights = list(tab_es.values())
tab_es_probab = [x/float(sum(tab_es_weights)) for x in tab_es_weights]
distr_es = pd.DataFrame({'states': tab_es_states, 'probab': tab_es_probab})
distr_es = distr_es.sort_values(by='probab', ascending=True)
distr_es.index = [i for i in range(len(tab_es_states))]


def huffman_branch(distr):
    states = list(distr['states'])
    probab = list(distr['probab'])
    state_new = [''.join(states[0:2])]
    probab_new = [probab[0] + probab[1]]
    codigo = list([{states[0]: 0, states[1]: 1}])
    states = states[2:] + state_new
    probab = probab[2:] + probab_new
    distr = pd.DataFrame({'states': states, 'probab': probab, })
    distr = distr.sort_values(by='probab', ascending=True)
    distr.index = [i for i in range(len(states))]
    branch = {'distr':distr, 'codigo':codigo}
    return(branch) 

def huffman_tree(distr):
    tree = []
    while len(distr) > 1:
        branch = huffman_branch(distr)
        distr = branch['distr']
        code = branch['codigo']
        tree += code
    return(tree)
 
tree_en = huffman_tree(distr_en)
tree_es = huffman_tree(distr_es)



'''
Funcion que crea un diccionario recorriendo las ramas y asignando 0 o 1 a cada 
letra segun la rama a la que pertenezca
'''
def huffman_dict(tree):
        dicc = {} 
        for c in list(tree[len(tree)-1].items())[0][0]:
            dicc[c] = "0"
        for c in list(tree[len(tree)-1].items())[1][0]:
            dicc[c] = "1"
        
        i = len(tree) - 2
        while i >= 0:
            for j in list(tree[i].items())[0][0]:
                dicc[j] += "0"
            for k in list(tree[i].items())[1][0]:
                dicc[k] += "1"
            i -= 1
                
        return(dicc)


dicc_en = huffman_dict(tree_en) 
dicc_es = huffman_dict(tree_es)


'''
Funcion que codifica una palabra dada en un idioma dado
'''
def codificar(palabra, idioma):
    codificacion = ""
    if(idioma == "I"):
        dicc = dicc_en
    if(idioma == "E"):
        dicc = dicc_es
    
    for c in palabra:
        codificacion += dicc[c]
        
    return(codificacion)


'''
Funcion que decodifica una palabra a partir de un codigo y un idioma dado
'''
def decodificar(codigo, idioma):
    decodificacion = ""
    if(idioma == "I"):
        dicc = {v:k for k, v in dicc_en.items()} 
    if(idioma == "E"):
        dicc = {v:k for k, v in dicc_es.items()} 
    
    aux = ""
    for c in codigo:
         aux += c
         if aux in dicc:
             decodificacion += dicc[aux]
             aux = ""
             
    return(decodificacion)
        


def apartado1():
    codif_en = codif_es = ""
    for c in en:
        codif_en += dicc_en[c]
    for c in es:
        codif_es += dicc_es[c]
    
    print("Código Huffman binario de S_en:")
    print(codif_en)
    print("Código Huffman binario de S_es:")
    print(codif_es)
    print("\n")
    
    long_en = long_es = 0
    for i in range(len(distr_en)):
        long_en += distr_en['probab'][i]*len(dicc_en[distr_en['states'][i]])
    for i in range(len(distr_es)):
        long_es += distr_es['probab'][i]*len(dicc_es[distr_es['states'][i]])
    
    print("Longitud media de S_en:")
    print(long_en)
    print("Longitud media de S_es:")
    print(long_es)
    print("\n")
    
    entr_en = entr_es = 0
    for i in range(len(distr_en)):
        entr_en -= distr_en['probab'][i]*math.log(distr_en['probab'][i], 2)
    for i in range(len(distr_es)):
        entr_es -= distr_es['probab'][i]*math.log(distr_es['probab'][i], 2) 
    
    print("Entropía de S_en:")
    print(entr_en)
    print("Entropía de S_es:")
    print(entr_es)
    print("\n")
    
    if(entr_en <= long_en < entr_en + 1):
        print("Se verifica el 1er Teorema de Shannon para S_en")
    else:
        print("NO se verifica el 1er Teorema de Shannon para S_en")
    if(entr_es <= long_es < entr_es + 1):
        print("Se verifica el 1er Teorema de Shannon para S_es")
    else:
        print("NO se verifica el 1er Teorema de Shannon para S_es")
        
        
def apartado2(palabra):
    codif_en = codificar(palabra, "I")
    codif_es = codificar(palabra, "E")
    print("Codificación de " + palabra + " en Inglés:")
    print(codif_en)
    print("Codificación de " + palabra + " en Español:")
    print(codif_es)
    
    
def apartado3(codigo):
    decodif_en = decodificar(codigo, "I")
    print("Decodificación de " + codigo + " en Inglés:")
    print(decodif_en)
    
    
apartado1()
print("\n")
print("---------------------")   
print("\n") 
apartado2("medieval")
print("\n")
print("---------------------")   
print("\n") 
apartado3("10111101101110110111011111")

  
    