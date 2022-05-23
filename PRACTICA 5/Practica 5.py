"""
DAVID SEIJAS PEREZ
PRACTICA 5
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

u = np.linspace(0.1, np.pi, 30) #0.1 para quitar el polo norte
v = np.linspace(0, 2*np.pi, 60)
x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))

#Curvas en esfera de radio 1
#Ecuador
t = np.linspace(0, 2*np.pi, 100)
x0 = np.sin(t)
y0 = np.cos(t)
z0 = np.zeros(len(t))

#Meridiano
x1 = np.zeros(len(t))
y1 = np.sin(t)
z1 = np.cos(t)

#Mi curva
t2 = np.linspace(0, 4*np.pi, 200)
x2 = 0.5*(1 + np.cos(t2))
y2 = 0.5*np.sin(t2)
z2 = np.sin(t2/2)


def proj(x, z, z0=1, alpha=1/2):
    z0 = z*0+z0
    eps = 1e-16
    return(x/(abs(z0-z)**alpha+eps))


def animate(t, eps = 1e-16):
    aux = 2/(2*(1-t) + (1-z)*t + eps)  #eps para no dividir por 0
    xt = aux*x 
    yt = aux*y
    zt = z*(1-t)-t
    
    ax = plt.axes(projection='3d')
    ax.set_zlim3d(-1,1)
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1, alpha=0.5, cmap='viridis', edgecolor='none')
    
    return ax,

def init():
    return animate(0),


def apartado1():
    Xp = proj(x,z)
    Yp = proj(y,z)
    
    #Dibujamos la esfera y la curva con sus proyeccion estereográficas
    fig = plt.figure(figsize=(15,15))
    fig.subplots_adjust(hspace=0.4, wspace=0.2)
    
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.plot(x0, y0, z0, '-b', zorder=3, label="Ecuador")
    ax.plot(x1, y1, z1, 'black', zorder=3, label="Meridiano")
    ax.plot(x2, y2, z2, '-r', zorder=3, label="Ejemplo")
    ax.set_title('2−sphere')
    
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.set_xlim3d(-1.5, 1.5)
    ax.set_ylim3d(-1.5, 1.5)
    ax.plot_surface(Xp, Yp, z*0, rstride=1, cstride=1, cmap='binary', edgecolor='black', linewidth=0.2)
    ax.plot(proj(x0, z0), proj(y0, z0), z0*0, '-b', zorder=3)
    ax.plot(proj(x1, z1), proj(y1, z1), z1*0, 'black', zorder=3)
    ax.plot(proj(x2, z2), proj(y2, z2), z2*0, '-r', zorder=3)
    ax.set_title('Proyección estereográfica')
    fig.legend(loc="upper center")
    
    plt.show()
    plt.close()
    
def apartado2():
    fig = plt.figure(figsize=(6,6))
    ani = animation.FuncAnimation(fig, animate, np.arange(0, 1, 0.05) , init_func=init, interval=30)
    ani.save("animacion.gif", fps=5)


apartado1()
apartado2()