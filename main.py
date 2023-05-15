import numpy as np
import time
from detect_faces import *
from feature_vector import * 

detect = detect()
feature = feature()

imagenes_prueba = [
            'foto_2.jpeg',
            'foto_3.jpeg', 'foto_4.jpeg',
            'foto_5.jpeg', 'foto_6.jpeg',
            'foto_7.jpeg', 'foto_8.jpeg',
            'foto_9.jpeg', 'foto_10.jpeg',
            'foto_11.jpeg'
        ]
start_time = time.time()

#Detectamos y guardamos los rostros de las imagenes de prueba
detect.detectAndSaveFaces('foto_1.jpeg')
for i in range(0, 10):
    detect.detectAndSaveFaces(imagenes_prueba[i])

print("--- %s segundos en total de deteccion de rostro usando CPU ---" % (time.time() - start_time))

faces = ['./rostros_detectados/foto_2.jpeg','./rostros_detectados/foto_3.jpeg','./rostros_detectados/foto_4.jpeg',
        './rostros_detectados/foto_5.jpeg','./rostros_detectados/foto_6.jpeg','./rostros_detectados/foto_7.jpeg',
        './rostros_detectados/foto_8.jpeg','./rostros_detectados/foto_9.jpeg','./rostros_detectados/foto_10.jpeg',
        './rostros_detectados/foto_11.jpeg'
        ]

imagenes_prueba = [
            './imagenes_prueba/foto_2.jpeg',
            './imagenes_prueba/foto_3.jpeg', './imagenes_prueba/foto_4.jpeg',
            './imagenes_prueba/foto_5.jpeg', './imagenes_prueba/foto_6.jpeg',
            './imagenes_prueba/foto_7.jpeg', './imagenes_prueba/foto_8.jpeg',
            './imagenes_prueba/foto_9.jpeg', './imagenes_prueba/foto_10.jpeg',
            './imagenes_prueba/foto_11.jpeg'
        ]
## Calculamos las distancias euclidianas con imagenes completas
aciertos = 0
start_time = time.time()

vect1 = np.array(feature.get_feature_vector('./imagenes_prueba/foto_1.jpeg'))
for i in range(0,10):
    vect2 = np.array(feature.get_feature_vector(imagenes_prueba[i]))
    dist = np.linalg.norm( vect1 - vect2)
    if(dist < 0.50):
        aciertos +=1

print("Usando imagen completa: %s segundos" % (time.time() - start_time))
porcentaje = aciertos / 10 * 100
print("Aciertos con imagenes completas: "+ str(porcentaje) + "%")

#Calculamos las distancias euclidianas con los rostros generados
aciertos = 0

start_time = time.time()
vect1 = np.array(feature.get_feature_vector('./rostros_detectados/foto_1.jpeg'))
for i in range(0,10):
    vect2 = np.array(feature.get_feature_vector(faces[i]))
    dist = np.linalg.norm( vect1 - vect2)
    if(dist < 0.50):
        aciertos +=1

print("Usando rostros: %s segundos" % (time.time() - start_time))
porcentaje = aciertos / 10 * 100
print("Aciertos con sólo rostros: "+ str(porcentaje) + "%")

