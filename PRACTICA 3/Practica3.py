'''
DAVID SEIJAS PEREZ
Practica 3
'''


import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


'''
Funcion que muestra los clusters calculados con KMEANS
'''
def mostrarKMEANS(X, labels, nClusters):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    
    #plt.figure(figsize=(8,4))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=5)
    
    plt.plot(problem[:,0],problem[:,1],'ko', markersize=10, markerfacecolor="green")
    plt.axis('tight')
    plt.title('Fixed number of KMeans clusters: %d' % nClusters)
    plt.show()


def apartado1():
    ks = [i for i in range(2, 16)]
    coefsSil = [0]*14
    
    centers = [[-0.5, 0.5], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,
                            random_state=0)
    
    #Algoritmo KMEANS para las distintas vencidades k
    for i in range(len(ks)):
        kmeans = KMeans(n_clusters=ks[i], random_state=0).fit(X)
        labels = kmeans.labels_
        silhouette = metrics.silhouette_score(X, labels)
        #mostrarKMEANS(X, labels, problem, ks[i])
        
        '''
        # Etiqueta de cada elemento (punto)
        print(kmeans.labels_)
        # Índice de los centros de vencindades o regiones de Voronoi para cada elemento (punto) 
        print(kmeans.cluster_centers_)
        #Coeficiente de Silhouette
        '''
        print("Silhouette Coefficient for k = " + str(ks[i]) 
              + ": %0.3f" % silhouette)
        coefsSil[i] = silhouette
        
    
    plt.ylim(0.3, 0.6)
    plt.plot(ks, coefsSil, 'ks-')
    plt.title('Coef Silhouette for each numer of clusters')
    plt.text(3.5, 0.58, "Máximo s")
    plt.show()
    #Con esto vemos que el mejor número de vecindades es k=3
    k = 3
    
    #Calculamos de nuevo todo para mostrar los clusters
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    labels = kmeans.labels_ 
    
    #Mostramos Diagrama de Vornoi junto con los clusters   
    vor = Voronoi(kmeans.cluster_centers_)
    voronoi_plot_2d(vor)
    mostrarKMEANS(X, labels, k)
    #Al quitar el plt.figure de mostrarClusters no nos elimina la figura de voronoi creada antes
    
    #Calculamos a qué clusters pertenecen los puntos a y b (Apartado 3)
    clases_pred = kmeans.predict(problem)
    print("Clasificación de los puntos a=(0,0) y b=(0,-1) para Kmeans:")
    print(clases_pred)
    print("\n------------------------------\n")
    
    

'''
Apartado 2
'''

'''
Funcion que muestra los cluster calculados con DBSCAN
'''
def mostrarDBSCAN(X, labels, core_samples_mask, nClusters):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    
    plt.figure(figsize=(8,4))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
    
        class_member_mask = (labels == k)
    
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=5)
    
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=3)
    
    plt.plot(problem[:,0],problem[:,1],'ko', markersize=10, markerfacecolor="green")
    plt.title('Estimated number of DBSCAN clusters: %d' % nClusters)
    plt.show()
    
    
def apartado2(distancia):
    epsilons = [0.1 + 0.005*i for i in range(60)]
    silMax = 0
    epsilonMax = 0.1

    centers = [[-0.5, 0.5], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,
                            random_state=0)
    
    #Algoritmo DBSCAN
    for i in range(len(epsilons)):
        db = DBSCAN(eps=epsilons[i], min_samples=10, metric=distancia).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        silhouette = metrics.silhouette_score(X, labels)  
        if(silhouette > silMax):
            silMax = silhouette
            epsilonMax = epsilons[i]
    
    
    db = DBSCAN(eps=epsilonMax, min_samples=10, metric=distancia).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
        
    print(distancia + " DISTANCE")
    print('Best epsilon: %0.3f' % epsilonMax)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % silMax)
    print("\n------------------------------\n")
        
    mostrarDBSCAN(X, labels, core_samples_mask, n_clusters_)
    

'''    
def apartado2():
    
    #Probar distintos epsilon en (0.1, 0.4) y analizar resultados
    #Desde 1.3 hasta 3 da 2 clusters cada vez con menos noisy points
    #Desde 3 a 4 da 1 cluster y con menos de 1.3 cada vez hay más clusters
    
    epsilon = 0.3
    centers = [[-0.5, 0.5], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,
                            random_state=0)
    
    #Algoritmo DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=10, metric='euclidean').fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print("EUCLIDEAN DISTANCE")
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    silhouette = metrics.silhouette_score(X, labels)
    print("Silhouette Coefficient: %0.3f"
          % silhouette)
    print("\n------------------------------\n")
    
    mostrarDBSCAN(X, labels, core_samples_mask, n_clusters_)
    
    
    Distancia Euclídea 
    ------------------
    Distancia Manhattan
    
    
    epsilon = 0.28
    centers = [[-0.5, 0.5], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4,
                            random_state=0)
    
    #Algoritmo DBSCAN
    db = DBSCAN(eps=epsilon, min_samples=10, metric='manhattan').fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print("MANHATTAN DISTANCE")
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    silhouette = metrics.silhouette_score(X, labels)
    print("Silhouette Coefficient: %0.3f"
          % silhouette)
    
    mostrarDBSCAN(X, labels, core_samples_mask, n_clusters_)
'''


problem = np.array([[-1.5, -1], [1.5, -1]])
apartado1()
#apartado2()
apartado2("euclidean")
apartado2("manhattan")