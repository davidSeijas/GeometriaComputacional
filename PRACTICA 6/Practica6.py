"""
DAVID SEIJAS PEREZ
PRACTICA 6
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib import animation


#q = variable de posición, dq0 = \dot{q}(0) = valor inicial de la derivada
#d = granularidad del parámetro temporal
def deriv(q, dq0, d):
    dq = (q[1:len(q)]-q[0:(len(q)-1)])/d
    dq = np.insert(dq, 0, dq0)  
    return dq


#Ecuación del sistema dinámico continuo
def F(q):
    ddq = -2*q*(q**2-1)
    return ddq


#Resolución de la ecuación dinámica \ddot{q} = F(q), obteniendo la órbita q(t)
#Los valores iniciales son la posición q0 := q(0) y la derivada dq0 := \dot{q}(0)
def orb(n, q0, dq0, F, args=None, d=0.001):
    q = np.empty([n+1])
    q[0] = q0
    q[1] = q0 + dq0*d
    for i in np.arange(2, n+1):
        args = q[i-2]
        q[i] = - q[i-2] + d**2*F(args) + 2*q[i-1]
    return q


## Pintamos el espacio de fases
def simplectica(q0, dq0, F, d, n, col=0, marker='-'):
    q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
    dq = deriv(q, dq0=dq0, d=d)
    p = dq/2
    plt.plot(q, p, marker, c=plt.get_cmap("winter")(col))


def diagrama_fases(t, d):
    q2 = np.array([])
    p2 = np.array([])
    
    #Condiciones iniciales
    seq_q0 = np.linspace(0.,1.,num=20)
    seq_dq0 = np.linspace(0.,2.,num=20)
    n = int(t/d) #t = n*delta
    
    for i in range(len(seq_q0)):
        for j in range(len(seq_dq0)):
            q0 = seq_q0[i]
            dq0 = seq_dq0[j]
            q = orb(n, q0=q0, dq0=dq0, F=F, d=d)
            dq = deriv(q, dq0=dq0, d=d)
            p = dq/2
            q2 = np.append(q2, q[-1])
            p2 = np.append(p2, p[-1])
            
    return (q2,p2)   


def animate(t):
    ax = plt.axes()
    (q,p) = diagrama_fases(t, d=10**(-4))
    plt.xlim(-2.5, 2.5)
    plt.ylim(-1.5, 1.5)
    #plt.plot(q, p, marker=".", markeredgecolor="black", markerfacecolor="blue")
    ax.scatter(q, p, c=q, cmap="winter", marker=".")
    return ax,


def init():
    return animate(0.1),


def apartado1():
    fig = plt.figure(figsize=(8, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    ax = fig.add_subplot(1, 1, 1)
    
    #Condiciones iniciales
    seq_q0 = np.linspace(0.,1.,num=12)
    seq_dq0 = np.linspace(0.,2.,num=12)
    d = 10**(-4)
    n = int(32/d)
    
    for i in range(len(seq_q0)):
        for j in range(len(seq_dq0)):
            q0 = seq_q0[i]
            dq0 = seq_dq0[j]
            col = (1+i+j*(len(seq_q0)))/(len(seq_q0)*len(seq_dq0))
            simplectica(q0=q0, dq0=dq0, F=F, col=col, marker=',', d=d, n=n)
    
    ax.set_xlabel("q(t)", fontsize=12)
    ax.set_ylabel("p(t)", fontsize=12)
    fig.savefig('Simplectic.png', dpi=250)
    plt.show()


def apartado2(t, d):
    fig = plt.figure(figsize=(8,5))
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    ax = fig.add_subplot(1,1,1)
    
    (q,p) = diagrama_fases(t, d=d)
    print("Calculando área de envoltura convexa para delta = ", d)
    
    ax.set_xlabel("q(t)", fontsize=12)
    ax.set_ylabel("p(t)", fontsize=12)
    plt.plot(q, p, marker="o", markersize= 10, markeredgecolor="black")
    plt.show()

    X = np.array([q,p]).T
    hull = ConvexHull(X)
    fig = convex_hull_plot_2d(hull)
    fig.savefig('Convexa.png', dpi=250)
    X_area = hull.volume
    print("Área total:", X_area)
    
    X_der = np.array([q[-20:], p[-20:]]).T
    hull_der = ConvexHull(X_der)
    #fig = convex_hull_plot_2d(hull_right)
    #fig.savefig('Convexa_derecha.png', dpi=250)
    X_der_area = hull_der.volume
    print("Área parte derecha:", X_der_area)
    
    X_inf = np.array([q[::20], p[::20]]).T
    hull_inf = ConvexHull(X_inf)
    #fig = convex_hull_plot_2d(hull_bottom)
    #fig.savefig('Convexa_inferior.png', dpi=250)
    X_inf_area = hull_inf.volume
    print("Área parte inferior:", X_inf_area)
    
    area = X_area - X_inf_area - X_der_area
    print("Área de D_t para t =", t, ":", area)
    

def apartado3():
    fig = plt.figure(figsize=(8,5))
    ani = animation.FuncAnimation(fig, animate, np.arange(0.2, 4.9, 0.1), init_func=init)
    ani.save("evolucion.gif", fps = 5)
    
#apartado1()
apartado2(0.25, 10**(-4))
apartado2(0.25, 10**(-3))
#apartado3()