# Laboratorio 2 Convolución y correlación
## Descripción
En este documento se busca reconocer los terminos de convolución y correlación, asi mismo saber en que momento se puede aplicar estas herramientas y en que señales o sistemas usarlas. También se define la transformada de Fourier y su importancia en el analisis del dominio del tiempo al de frecuencia.
## Convolución
La convolución es una herramienta matemática que describe el proceso de "deslizar" una función sobre otra. En términos más específicos, se refiere a multiplicar los valores de una función por los valores de la otra en los puntos de superposición y luego sumar esos productos para generar una nueva función.

$y[n] = x[n] * h[n] = \sum_{k=-\infty}^{\infty} x[k] h[n-k]$

En el contexto del procesamiento digital de señales, la convolución se emplea para estudiar y diseñar sistemas LTI (Lineales y de Tiempo Invariante), como los filtros digitales. La salida de un sistema LTI, denotada como $y[n]$, se obtiene mediante la convolución de la señal de entrada $x[n]$ con la respuesta al impulso $h[n]$, de acuerdo con la siguiente ecuación:

$y[n]= x[n]*h[n]$

Para poder hacer una convolución en pyhton se pueden seguir los siguientes pasos:

### Librerias
```python
import matplotlib.pyplot as plt
import numpy as np
import wfdb
import seaborn as sns
from scipy.fftpack import fft
from scipy.signal import welch
```
### Señal de entrada y de salida (explicar np.convolve)
```python
codigo_Youling=np.array([5,6,0,0,8,1,5])
codigo_Jose=np.array([5,6,0,0,7,9,3])
codigo_Camilo=np.array([5,6,0,0,7,4,5])

cedula_Youling=np.array([1,0,1,2,3,4,1,6,5,2])
cedula_Jose=np.array([1,0,9,3,4,3,2,8,3,9])
cedula_Camilo=np.array([1,0,1,1,0,8,5,8,8,0])


conv_Youling=np.convolve(codigo_Youling, cedula_Youling, mode='full')
conv_Jose=np.convolve(codigo_Jose, cedula_Jose, mode='full')
conv_Camilo=np.convolve(codigo_Camilo, cedula_Camilo, mode='full')
```
[![Datos-convoluci-n.jpg](https://i.postimg.cc/kgQBZKCY/Datos-convoluci-n.jpg)](https://postimg.cc/xXdfNkcL)
### Graficar la señal
````python
plt.figure(figsize=(10,6))
plt.stem(conv_Youling)
plt.title('Convolución Youling')
plt.xlabel('Datos')
plt.ylabel('Y(n)')
plt.grid()
plt.show()
````
[![Convoluci-n-Youling.jpg](https://i.postimg.cc/zB6h5Ncx/Convoluci-n-Youling.jpg)](https://postimg.cc/TLj1qzGb)
[![Convoluci-n-Jos.jpg](https://i.postimg.cc/FHNLhDTX/Convoluci-n-Jos.jpg)](https://postimg.cc/ZCsRcFkw)
[![Convoluci-n-Camilo.jpg](https://i.postimg.cc/SKt27qkm/Convoluci-n-Camilo.jpg)](https://postimg.cc/cKwL0pvz)


## Correlación


## Electromiografía
### Caracterización
### Clasificación
### Transformada de Fourier
### Análisis estadísticos descriptivos
### Bibliografía
(S/f). Mathworks.com. Recuperado el 11 de febrero de 2025, de https://la.mathworks.com/discovery/convolution.html
### Colaboradores
- Youling Andrea Orjuela Bermúdez (5600815)
- José Manuel Gomez Carrillo (5600793)
- Juan Camilo Quintero Velandia (5600745)

