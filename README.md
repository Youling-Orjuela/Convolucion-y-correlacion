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
La correlación de señales es una herramienta matemática que permite medir la similitud entre dos señales. Esto es útil para determinar qué tan parecidas son las señales, tanto en términos de su forma como de su comportamiento a lo largo del tiempo.

Además de comparar señales, la correlación también permite detectar patrones periódicos (periodicidad), lo que es útil para identificar señales que se repiten con regularidad. Por otro lado, la correlación cruzada se utiliza para la detección de retardos temporales entre señales y para la sincronización de señales relacionadas.

La fórmula general de la correlación cruzada entre dos señales $x[n]$ y $y[n]$ es:

$R_{xy}[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot y[k+n]$

SIguiendo el siguiente esquema se puede obtener la correlación de estas dos señales en python

𝑥1[𝑛𝑇𝑠] = cos(2𝜋100𝑛𝑇𝑠) 𝑝𝑎𝑟𝑎 0 ≤ 𝑛 < 9

𝑥2[𝑛𝑇𝑠] = sin(2𝜋100𝑛𝑇𝑠) 𝑝𝑎𝑟𝑎 0 ≤ 𝑛 < 9 𝑝𝑎𝑟𝑎 𝑇𝑠 = 1.25𝑚s

### Definir la señal
````python
# Parámetros
fs = 1 / (1.25e-3)  # Frecuencia de muestreo
f = 100  # Frecuencia de la señal
Ts = 1.25e-3  # Período de muestreo
n = np.arange(0, 9)  # Rango de n
t = n * Ts  # Tiempo discreto

# Definición de las señales
x1 = np.cos(2 * np.pi * f * t)
x2 = np.sin(2 * np.pi * f * t)

# Cálculo de la correlación cruzada
correlacion = np.correlate(x1, x2, mode='full')
lags = np.arange(-len(n) + 1, len(n))
````
### Graficar la señal y correlación cruzada
````python
# Gráficos
plt.figure(figsize=(12, 5))

# Señales
plt.subplot(2, 1, 1)
plt.stem(n, x1, 'b', markerfmt='bo', label='$x_1[n] = \cos(2\pi100nT_s)$')
plt.stem(n, x2, 'r', markerfmt='ro', label='$x_2[n] = \sin(2\pi100nT_s)$')
plt.xlabel('n')
plt.ylabel('Amplitud')
plt.title('Señales $x_1[n]$ y $x_2[n]$')
plt.legend()
plt.grid()

# Correlación cruzada
plt.subplot(2, 1, 2)
plt.stem(lags, correlacion, 'g', markerfmt='go', label='Correlación cruzada')
plt.xlabel('Desplazamiento (k)')
plt.ylabel('Amplitud')
plt.title('Correlación cruzada entre $x_1[n]$ y $x_2[n]$')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
````
[![Correlaci-n.jpg](https://i.postimg.cc/DZR8bgF6/Correlaci-n.jpg)](https://postimg.cc/VrBYxXJ0)


## Electromiografía
La electromiografia es el examen que nos permite determinar el funcionamiento de los musculos. En la plataforma Physionet, dentro de la sección de Data Open Access, se puede acceder a diversos conjuntos de datos relacionados con señales biomédicas, incluyendo señales de electromiografía (EMG).

Para obtener una señal de EMG, se selecciona un conjunto de datos disponible en esta sección. Luego, se descargan los archivos correspondientes: los archivos con extensión .hea y .dat. Estos archivos contienen la información necesaria para realizar los diferentes procedimientos de análisis, que se verán a continuación:

### Extraer y graficar la señal
````python
# Nombre del archivo del registro
electromiografia = 'emg_healthy'

try:
    # Leer el archivo del registro
    record = wfdb.rdrecord(electromiografia)
except FileNotFoundError:
    print(f"Archivo '{electromiografia}' no encontrado.")
    exit()

# Extraer la señal, etiquetas, frecuencia de muestreo y tiempo
senal = record.p_signal
etiquetas = record.sig_name
frecuencia = record.fs  # Frecuencia de muestreo
tiempo = np.arange(senal.shape[0]) / frecuencia  # Crear vector de tiempo
````
### Graficar la señal de electromiografia
````python
# Información básica
print(f"Frecuencia de muestreo: {frecuencia} Hz")
print(f"Forma completa de la señal: {senal.shape}")

plt.figure(figsize=(12, 5))
plt.title("Señal Electromiográfica")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (mV)")
plt.plot(tiempo, senal[:, 0], label='Electromiografía', color='purple')
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
````
[![EMG.jpg](https://i.postimg.cc/7ZVvGZDQ/EMG.jpg)](https://postimg.cc/4Ym2jZYz)
#### Información básica
[![FM.jpg](https://i.postimg.cc/BvLgWmhc/FM.jpg)](https://postimg.cc/McwRVmJv)

### Caracterización
#### Estadisticos descriptivos
````python
# Parte de estadística con señal completa
print('Datos estadísticos con el empleo de funciones: ')
print('La media es: ', np.mean(senal))
print('La desviación estándar es: ', np.std(senal))
print('El coeficiente de variación es: ', (np.std(senal) / np.mean(senal)))
````
[![Datos-estadisticos.jpg](https://i.postimg.cc/nVkqW8Ff/Datos-estadisticos.jpg)](https://postimg.cc/kD2VVpRT)

### Clasificación
#### Histograma
````python
# Histograma de amplitud
plt.figure(figsize=(12, 6))
sns.histplot(senal[:, 0], bins=50, color='blue', edgecolor='black', alpha=0.7, kde=True)
plt.title('Histograma de la Amplitud de la Señal Electromiográfica')
plt.xlabel('Amplitud (mV)')
plt.ylabel('Frecuencia')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
````
[![Histograma.jpg](https://i.postimg.cc/DwyJH5vj/Histograma.jpg)](https://postimg.cc/svqgZYZ7)

### Transformada de Fourier
#### Análisis espectral y graficar la transformada de Fourier
````python
#Analisis espectral
n = len(senal[:, 0])
freqs = np.fft.fftfreq(n, d=1/frecuencia)
fft_vals = np.abs(fft(senal[:, 0])) / n

# Graficar Transformada de Fourier
plt.figure(figsize=(12, 5))
plt.plot(freqs[freqs >= 0], fft_vals[freqs >= 0], color='red')
plt.title("Transformada de Fourier de la Señal")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlim(0, 500)
plt.show()
````
[![Transformada.jpg](https://i.postimg.cc/4yWZ96Dy/Transformada.jpg)](https://postimg.cc/SYYw0MNb)
#### Densidad espectral de potencia
````python
# Calcular y graficar la Densidad Espectral de Potencia (PSD)
freqs_psd, psd = welch(senal[:, 0], fs=frecuencia, nperseg=1024)
plt.figure(figsize=(12, 5))
plt.semilogy(freqs_psd, psd, color='green')
plt.title("Densidad Espectral de Potencia (PSD)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Densidad de Potencia (dB/Hz)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.xlim(0, 500)
plt.show()
````
[![Densidad-espectral.jpg](https://i.postimg.cc/tJTY14s3/Densidad-espectral.jpg)](https://postimg.cc/7GFxdDMh)

#### Estadísticos descriptivos en función de la frecuencia
````python
freqs_pos = freqs[freqs >= 0]
fft_magnitude_pos = fft_vals[freqs >= 0]
freq_mean = np.sum(freqs_pos * fft_magnitude_pos) / np.sum(fft_magnitude_pos)  # Frecuencia media
cumsum = np.cumsum(fft_magnitude_pos)
freq_median = freqs_pos[np.where(cumsum >= cumsum[-1]/2)[0][0]]  # Frecuencia mediana
freq_std = np.sqrt(np.sum(fft_magnitude_pos * (freqs_pos - freq_mean) ** 2) / np.sum(fft_magnitude_pos))  # Desviación estándar
````
[![Estadistico-en-frecuencia.jpg](https://i.postimg.cc/RVWW4q59/Estadistico-en-frecuencia.jpg)](https://postimg.cc/TpXdcd3H)
#### Histograma de frecuencias
````python
plt.figure(figsize=(12, 6))
plt.hist(freqs_pos, bins=100, weights=fft_magnitude_pos, edgecolor='black', alpha=0.7, color='yellow')
plt.title('Histograma de Frecuencias de la Señal Electromiográfica')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Densidad')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(0, 500)
plt.show()
````
[![Histograma-frecuencia.jpg](https://i.postimg.cc/HWhLZHJN/Histograma-frecuencia.jpg)](https://postimg.cc/ThnG1zJj)

### Análisis estadísticos descriptivos

Para caracterizar la señal electromiográfica (EMG) se hizo uso de las siguientes funciones para determinar los estadisticos correspondientes a la misma:

En primer lugar, la frecuencia de muestreo, se extrae por medio de la función record.fs, la cual nos indica que el muestreo se esta realizando a 4000 Hz. En cuanto a los estadísticos descriptivos, el valor de la media; el cual se extrae con la funcion np.mean(senal), da como resultado un valor, producto del análisis de los diferentes valores de amplitud que toma la señal en su captura, siendo 0,000199 el valor de voltaje medio de la señal. De este valor se puede analizar que su magnitud es muy cercana a cero gracias a que la señal tiene trazos positivos y negativos.

Por otro lado, el valor de desviación estándar; el cual tiene como resultado un valor de aproximadamente 0,08157 y cuyo número es extraido con la funcion np.std(senal), sugiere que la desviacion de los datos en la funcion no se encuentran muy dispersos respecto a la media de valores y que a nivel general, la actividad muscular es moderada. 
 
Por ultimo, la del coeficiente de variación, el cual es extraído producto de la razón entre la desviación estándar y la media de la señal y que toma un valor cercano a 408,225, indica que pese a que la desviación es baja, la señal puede experimentar picos muy elevados de manera esporádica y durante toda la captura de la señal misma.

### Bibliografía
(S/f). Mathworks.com. Recuperado el 11 de febrero de 2025, de https://la.mathworks.com/discovery/convolution.html

Palacios, J. (s/f). Sección 2.7 Correlación de señales discretas en el tiempo. SlideShare. Recuperado el 11 de febrero de 2025, de https://es.slideshare.net/slideshow/seccin-27-correlacin-de-seales-discretas-en-el-tiempo/234297001

De vista temporal, E. E. T. se A. las S. y. S. D. D. el P., & de trabajar con dichos sistemas., S. C. B. E. I. a. la H. (s/f). OBJETIVOS DEL TEMA. Www.uv.es. Recuperado el 11 de febrero de 2025, de https://www.uv.es/soriae/tema_2_pds.pdf



### Colaboradores
- Youling Andrea Orjuela Bermúdez (5600815)
- José Manuel Gomez Carrillo (5600793)
- Juan Camilo Quintero Velandia (5600745)

