#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 00:21:45 2021

@author: sifoline
"""

from TrgPO import Triangle, Billiard

import numpy as np
import matplotlib.pyplot as plt # Temporary solution

from numpy import pi, cos, sin, tan, floor, ceil, exp, arccos, arcsin, arctan

import numexpr as ne

# Numerical Tolerance
NumTol = 1e-6

##### 
M = Triangle(pi/4,pi/4)

x = Billiard(1.5,1, M, Iter = 500)


TLog,PtLog,Periodic = x.Evolve()


# Public Methods Cont'd

M.Plot()
plt.plot(PtLog[:,0],PtLog[:,1],alpha = 1)
            
Fig2 = plt.figure()

ax2 = Fig2.add_subplot()

ax2.scatter(TLog[:,0],TLog[:,1],color = 'black',s = 0.7)
ax2.scatter(TLog[0,0],TLog[0,1],color = 'red')

ax2.set_xlim([0,3])
ax2.set_ylim([0,pi])

ax2.hlines(pi/2,0,3,linestyles ='dashed')
