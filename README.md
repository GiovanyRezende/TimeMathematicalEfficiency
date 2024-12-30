# Introduction
*This is a simple self-research project on how mathematical simplifications can optimize time in a project.* Several fields in the technology industry directly need Mathematics, such as Data Science and Robotics. However, with the quantity of libraries and frameworks with prebuilt mathematical functions and methods, it seems like a data scientist doesn't need to study the mathematics of Machine Learning, for example. The research was partitioned into two stages: 

- The first one works with an arbitrary function. The proposal is to compare the time to integrate this function with quadrature formula and with the its actual integral;
- The second one also is a comparison between two integral methods, but with the cumulative probability of a normal distribution.

This research was made in Python 3.

# Libraries
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import e
from scipy.integrate import quad
import scipy.special as sp
from time import time
```

# Stage 1

## The arbitrary function
The chosen function for the research is:

$$e^{2x} + e^{\frac{x}{2}} + e^{-2x} + e^{-\frac{x}{2}}$$

Its defined integral is:

$$
\int_a^b \left( e^{2x} + e^{\frac{x}{2}} + e^{-2x} + e^{-\frac{x}{2}} \right) dx = 
\left[ \frac{1}{2}(e^{2x} - e^{-2x}) + 2(e^{\frac{x}{2}} - e^{-\frac{x}{2}}) \right]_a^b
$$

The code for the function and its defined integral is:
```
def f(x):
  return e**(2*x) + e**(x/2) + e**(-2*x) + e**(-x/2)

def integral_f(a,b):
  x1 = 0.5*(e**(2*a) - e**(-2*a)) + 2*(e**(a/2) - e**(-a/2))
  x2 = 0.5*(e**(2*b) - e**(-2*b)) + 2*(e**(b/2) - e**(-b/2))
  return x2 - x1
```

The defined integral with quadrature formula is made with ```scipy.integrate``` method ```quad```.

## Data extraction
The defined integral was made from 0 to all numbers in ```np.arange(0.1,100.1,0.1)```. So:

```
numbers = np.arange(0.1,100.1,0.1)
time_scipy = []
time_integral = []

for i in numbers:
  start = time()
  quad(f,0,i)
  end = time()
  time_scipy.append(end-start)

for i in numbers:
  start = time()
  integral_f(0,i)
  end = time()
  time_integral.append(end-start)

dict = {
    'time_scipy': time_scipy,
    'time_integral': time_integral
}

df = pd.DataFrame(dict)
```

This code iterates at ```numbers``` array and calculates the defined integral for each element, with the two forms of integration. The time interval is measured with ```time()``` difference between before the integral computing and after the computing. The data was stored in a dictionary, and then converted to a Pandas dataframe.

## Data visualization
```
plt.style.use('Solarize_Light2')
plt.scatter(numbers,df['time_scipy'],color='r')
plt.scatter(numbers,df['time_integral'],color='b')
plt.xlabel('Input Number')
plt.ylabel('Time')
plt.show()
```

The Scipy quadrature formula is symbolized by red color and the previously calculated integral is symbolized by blue color:

![image](https://github.com/user-attachments/assets/fdf58240-37df-4dad-927b-5abe02278643)

## Observation
The time interval measured can seem to be random from experiment to experiment! The same experiment was made again:

```
time_scipy_2 = []
time_integral_2 = []

for i in numbers:
  start = time()
  quad(f,0,i)
  end = time()
  time_scipy_2.append(end-start)

for i in numbers:
  start = time()
  integral_f(0,i)
  end = time()
  time_integral_2.append(end-start)

dict_2 = {
    'time_scipy': time_scipy_2,
    'time_integral': time_integral_2
}

df_2 = pd.DataFrame(dict_2)
```
```
plt.style.use('Solarize_Light2')
plt.scatter(numbers,df_2['time_scipy'],color='r')
plt.scatter(numbers,df_2['time_integral'],color='b')
plt.xlabel('Input Number')
plt.ylabel('Time')
plt.show()
```
![image](https://github.com/user-attachments/assets/389f96c9-59be-43a9-95cf-77e9710c3e09)

```
time_scipy_3 = []
time_integral_3 = []

for i in numbers:
  start = time()
  integral_f(0,i)
  end = time()
  time_integral_3.append(end-start)

for i in numbers:
  start = time()
  quad(f,0,i)
  end = time()
  time_scipy_3.append(end-start)

dict_3 = {
    'time_scipy': time_scipy_3,
    'time_integral': time_integral_3
}

df_3 = pd.DataFrame(dict_3)
```
```
plt.style.use('Solarize_Light2')
plt.scatter(numbers,df_3['time_scipy'],color='r')
plt.scatter(numbers,df_3['time_integral'],color='b')
plt.xlabel('Input Number')
plt.ylabel('Time')
plt.show()
```
![image](https://github.com/user-attachments/assets/8a9bd824-6437-49d1-ad4b-33df5253ccad)

Doing the experiment several times can return different results from each other. Overall, *the previously calculated integral had better results in time optimizing*.

# Stage 2

## The Cumulative Normal Distribution Function computing
The normal distribution is:

$$
f(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
$$

Its defined integral, the cumulative distribution, can be simplified as:

$$
F(x) = \frac{1}{2} \left( 1 + \{erf}\left( \frac{x - \mu}{\sigma \sqrt{2}} \right) \right)
$$

$$erf$$ is the error function:

$$
\{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \, dt
$$

The error function doesn't have an easy solution, so ```scipy.special``` was used. For the code:

```
def standard_normal(x,u=5,o=2):
    return (1/(o*np.sqrt(2*np.pi))) * np.exp(-(((x-u)/o)**2)/2)

def cndf(x,u=5,o=2):
    return 0.5 * (1 + sp.erf(((x-u)/o)/ np.sqrt(2)))
```

## Data extraction and visualization
This stage is similar to the previous one, except for the fact that were used four scatter graphics to study the computing for different ranges:

```
plt.style.use('Solarize_Light2')
fig, ax = plt.subplots(2, 2, figsize=(10, 8))

plot = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

for i in range(0,4):
  time_scipy = []
  time_erf = []
  numbers = np.arange(0.1, (10**(i + 1)) + 0.1, 0.1)

  for j in numbers:
    start = time()
    quad(standard_normal,-np.inf,i)
    end = time()
    time_scipy.append(end-start)

  for j in numbers:
    start = time()
    cndf(i)
    end = time()
    time_erf.append(end-start)

  dict_cdf = {
      'time_scipy': time_scipy,
      'time_erf': time_erf
  }

  df_cdf = pd.DataFrame(dict_cdf)


  ax[plot[i][0],plot[i][1]].scatter(numbers,df_cdf['time_scipy'],color='r')
  ax[plot[i][0],plot[i][1]].scatter(numbers,df_cdf['time_erf'],color='b')
  ax[plot[i][0],plot[i][1]].set_xlabel('Input Number')
  ax[plot[i][0],plot[i][1]].set_ylabel('Time')

plt.tight_layout()
plt.show()
```

The graphics are:
![image](https://github.com/user-attachments/assets/47b23d2d-8b91-4b3e-ab30-7879e90cd2c0)

# Conclusion
Although time interval discrepancy doesn't seem to be significant (while none of the computings took a long time), the data visualization expresses how mathematical simplification is more efficient and cohesive. This can make a difference in Big Data context, for example. That's why it's important to understand how to make mathematical simplifications!

<div align= center>

# Contact



[![logo](https://cdn-icons-png.flaticon.com/256/174/174857.png)](https://br.linkedin.com/in/giovanyrezende)
[![logo](https://images.crunchbase.com/image/upload/c_lpad,f_auto,q_auto:eco,dpr_1/v1426048404/y4lxnqcngh5dvoaz06as.png)](https://github.com/GiovanyRezende)[
![logo](https://logospng.org/download/gmail/logo-gmail-256.png)](mailto:giovanyrmedeiros@gmail.com)

</div>
