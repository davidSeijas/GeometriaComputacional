"""
Practica 1
DAVID SEIJAS
"""

import matplotlib.pyplot as plt
import random



def logistica(x, r):
    return r*x*(1-x);


'''
def fn(x0, f, n):
    x = x0
    for j in range(n):
        x = f(x)
    return x
'''


def orbita(r, x0, f, N):
    orb = [0]*N
    orb[0] = x0
    for i in range(1, N):
        orb[i] = f(orb[i-1], r)
    return orb


def periodo(suborb, epsilon=0.001):
    N = len(suborb)
    for i in range(2, N+1):
        if abs(suborb[N-1] - suborb[N-i]) < epsilon :
            break
    return i-1
    

def atrac1(f, r, x0, N0, N, epsilon=0.001):
    orb = orbita(r, x0, f, N0)
    ult = orb[N0-N-1:]
    per = periodo(ult, epsilon)
    atractor = [0]*per
    anterior = [0]*per
    for i in range(per):
        atractor[i] = ult[N-1-i]
        anterior[i] = ult[N-1-i-per]
    return (atractor, anterior)
#Los primeros per elems son los atractores y los ultimos per elems son los 
#puntos anteriores a esos atractores en la curva que tiende a la asintota

def atrac2(f, r, x0, N0, N, epsilon=0.001):
    orb = orbita(r, x0, f, N0)
    ult = orb[N0-N-1:]
    per = periodo(ult, epsilon)
    atractor = [0]*per
    for i in range(per):
        atractor[i] = ult[N-1-i]
    return atractor


#Apartado 1
def conjAtractor():
    k = 2
    for i in range(k):
        r = random.uniform(3.0, 3.544)
        x0 = random.uniform(0.0, 1.0)
        atractor, anterior = atrac1(logistica, r, x0, N0, N)
        print("Conjunto atractor para r = " + str(r) + " y x0 = " + str(x0) + ":")
        for j in range(len(atractor)):
            print(str(atractor[j]) + " +/- " + str(abs(atractor[j] - anterior[j])))
        graficos1(r, x0, N0)


def graficos1(r, x0, N0):
    orb = orbita(r, x0, logistica, N0)
    plt.title('r = ' + str(r))
    plt.plot(orb)
    plt.show()

'''
def graficos1(rs, x0s, N0):
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    for i in range(3):
        for j in range(2):
            r = rs[i*2+j]
            x0 = x0s[i*2+j]
            axs[i, j].plot(orbita(r, x0, logistica, N0))
            axs[i, j].title.set_text('r = ' + str(r))
    
    plt.show()
'''


#Apartado 2
def atractor8elems(tam = 8):
    paso = 0.0006
    error = 0.0003
    time_trans = 1000
    
    rs = [3.4000 + paso*i for i in range(time_trans)] #(3.4000, 4.0000)
    x0 = random.uniform(0.0, 1.0)
    inicio = 0
    fin = 0
    atractores = [0]*time_trans
    
    for i in range(time_trans):
        atractor = atrac2(logistica, rs[i], x0, N0, N)
        if len(atractor) == tam and inicio == 0:
            inicio = (rs[i] + rs[i-1])/2
        if len(atractor) > tam and fin == 0:
            fin = (rs[i] + rs[i-1])/2
        atractores[i] = atractor + [float("nan")]*(N-len(atractor))

    print("Conjunto atractor con 8 elems en intervalo de r:")
    print("[" + str(inicio) + " +/- " + str(error) + ", " + str(fin) + " +/- " + str(error) + "]")
    graficos2(rs, atractores, inicio, fin)
    
    #Coger el ultimo con 4 elems y el primero con 8 y cogemos punto medio 
    #+/- la mitad del paso (0.0005). Idem para el otro extremo
    

def graficos2(r, atractores, int_min, int_max):
    plt.figure(figsize=(10,10))
    plt.plot(r, atractores, 'ro', markersize=1)  
    plt.xlabel = "r"
    plt.ylabel = "atractores"
    plt.axvline(x=int_min, ls="--")
    plt.axvline(x=int_max, ls="--")
    plt.show()

    
N0 = 200
N = 50
conjAtractor()
atractor8elems()

