# ----------------------------
# ZADANIE 1
# ----------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

T = 2
w=2*np.pi/T
def f(t):
  return 1.4*t**2

#a0 ------------------
def ia0(t):
  return f(t)
a0 = 1/T*quad(ia0, -1, 1)[0]

#a1 ------------------
def ia1(t):
  return f(t)*np.cos(np.pi*t)
a1 = 2/T*quad(ia1, -1, 1)[0]

#a2 ------------------
def ia2(t):
  return f(t)*np.cos(2*np.pi*t)
a2 = 2/T*quad(ia2, -1, 1)[0]

#b1 ------------------
def ib1(t):
  return f(t)*np.sin(np.pi*t)
b1 = 2/T*quad(ib1, -1, 1)[0]

#b1 ------------------
def ib2(t):
  return f(t)*np.sin(2*np.pi*t)
b2 = 2/T*quad(ib2, -1, 1)[0]
 

# F(t)
def F(t):
  return a0+a1*np.cos(np.pi*t)+a2*np.cos(2*np.pi*t)+b1*np.cos(np.pi*t)+b2*np.cos(2*np.pi*t)

print("Wsp:")
print("a0:", a0)
print("a1:", a1)
print("a2:", a2)
print("b1:", b1)
print("b2:", b2)
print()
 
t=np.linspace(-1,1,100)
fig, ax = plt.subplots()
ax.plot(t,f(t),t,F(t))
ax.grid(True)
plt.show() 


# ----------------------------
# ZADANIE 2
# ----------------------------
import numpy as np
from sympy import * 
a,b,c,t = symbols('a b c t') 
A,B,C=0.4666666666666666, -0.5673986283970915, 0.14184965709927283
w=pi
expr=(a*t**2+b*t+c -(A+B*cos(w*t)+C*cos(2*w*t)))*t**2
sol1=integrate(expr, (t,-1,1))
expr=(a*t**2+b*t+c -(A+B*cos(w*t)+C*cos(2*w*t)))*t
sol2=integrate(expr, (t,-1,1))
expr=(a*t**2+b*t+c -(A+B*cos(w*t)+C*cos(2*w*t)))
sol3=integrate(expr, (t,-1,1))
 
equations = [
 Eq(sol1, 0 ),
 Eq( sol2, 0 ),
 Eq(sol3,0)
]
print(sol1)
print("----------------------")
print(sol2)
print("----------------------")
print(sol3)
print("----------------------")
print (solve(equations))

# ----------------------------
# ZADANIE 3
# ----------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def f(t):
  x=(t-1)%T-1
  return 1.4*x*x

 
t=np.linspace(-1,5,100)
fig, ax = plt.subplots()
ax.plot(t,f(t))
ax.grid(True)
plt.show() 

