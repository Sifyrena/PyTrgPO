#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:51:49 2021

@author: ywan598
"""

import numpy as np
import matplotlib.pyplot as plt # Temporary solution

from numpy import pi, cos, sin, tan, floor, ceil, exp, arccos, arcsin, arctan


################ BREAD ################

# Triangle Definition Inherited from MATLAB Version. A: (0,0), B: (1,0)


class Triangle:
        


    def __init__(self,AnA,AnB):
        
        self.AnA = AnA
        self.AnB = AnB
        self.AnC = pi - AnA - AnB
        
        self.PtA = np.array([0,0],dtype = 'float64')
        self.PtB = np.array([1,0],dtype = 'float64')
        self.PtC = self.PtC_Calc()
        
        self.Circum, self.ProjB, self.ProjR, self.ProjL = self.Circumference()
        
        self.Shape = {'B': [self.PtA,self.PtB],
                      'R': [self.PtB,self.PtC],
                      'L': [self.PtC,self.PtA]
            }
        
        self.Edges = ["B","R","L","B"]
        
        
            
    def PtC_Calc(self):
        
        AnA = self.AnA
        AnB = self.AnB
        
        PtCx = tan(AnB)/(tan(AnA)+tan(AnB))
        PtCy = tan(AnA)*tan(AnB)/(tan(AnA)+tan(AnB)) 
        return np.array([PtCx,PtCy])

    def CoM(self):
        return (self.PtB + self.PtC)/2


    def Circumference(self):
        
        d_AC = self.PtC - self.PtA
        L_AC = np.linalg.norm(d_AC)
        
        d_BC = self.PtC - self.PtB
        L_BC = np.linalg.norm(d_BC)
        
        return L_AC+L_BC+1, 1, L_BC, L_AC
    

    def Plot(self):
        
        Vertex = np.array([self.PtA,self.PtB,self.PtC,self.PtA])
                
        fig = plt.figure()
        
        ax = fig.add_subplot(111)
        print(Vertex)
        
        ax.plot(Vertex[:,0],Vertex[:,1])
        ax.set_aspect('equal')
        ax.grid()

################ BUTTER ################


class Billiard:
    
    def __init__(self,c0,Th0, Triangle, Iter = 4):
        
            
        self.Iter = Iter
        
        self.Triangle = Triangle
        
        # Important and dynamic
        
        if c0 < 0 or c0>= 3:
            raise ValueError('The Initial Projected Location Must Be Between [0 and 3)!')
            
        elif c0 == np.floor(c0):
            raise ValueError('Cannot Start on Vertex')
            
        else:
            self.c = c0
        
        if (Th0 <= 0) or (Th0 >= pi):
            raise ValueError('The Initial Angle Must Be Between (0 and pi) radians!')
        else:
            self.Th = Th0
            
        self.CurrentEdge, self.ProjDist = self.WhereAmI()
        
    def WhereAmI(self):
        
        c = self.c
        
        if floor(c) == 0:
            return 'B', c - floor(c)
        
        if floor(c) == 1:
            return 'R', c - floor(c)
        
        if floor(c) == 2:
            return 'L', c - floor(c)


    def TranslateLocation(self): # Converts Billird c value to Euclidean Position
        c = self.c    
    
        if floor(c) == 0:
            # On AB
            DWt = c-floor(c)

            PtStart = self.Triangle.PtA
            PtEnd = self.Triangle.PtB
                      
        
        if floor(c) == 1:
            # On BC
            DWt = c-floor(c)
            
            PtStart = self.Triangle.PtB
            PtEnd = self.Triangle.PtC
            
        
        if floor(c) == 2:
            # On CA
            DWt = c-floor(c)
            
            PtStart = self.Triangle.PtC
            PtEnd = self.Triangle.PtA

        
        return PtStart*(1-DWt) + PtEnd*(DWt)
        
    
    
    def TranslateC(self,Edge,Point): # Converts Billird c value to Euclidean Position
        
    
        if Edge == 'B':
            # On AB
            
            PtStart = self.Triangle.PtA
            PtEnd = self.Triangle.PtB
            PreAdd = 0
        
        if Edge == 'R':
            # On BC

            PtStart = self.Triangle.PtB
            PtEnd = self.Triangle.PtC
            PreAdd = 1
            
        
        if Edge == 'L':
            # On CA

            PtStart = self.Triangle.PtC
            PtEnd = self.Triangle.PtA
            PreAdd = 2


        return ((Point[0]-PtStart[0])/(PtEnd[0]-PtStart[0])) + PreAdd
    
    
    
        
    def CriticalAngles(self):
        
        Edge = self.CurrentEdge
        Pos = self.TranslateLocation()

        Vertex = self.Triangle.Shape[Edge]
        
        if Edge == 'B':
            PtOff = self.Triangle.PtC
        
        
        if Edge == 'R':
            PtOff = self.Triangle.PtA
            
            
        if Edge == 'L':
            PtOff = self.Triangle.PtB
            
        
        EdgeVec = Vertex[1] - Vertex[0] # Kept Chirality
        EdgeVec /= np.linalg.norm(EdgeVec)
        
        if Edge == 'L':
            EdgeVec *= -1
        
        
        Relative = PtOff - Pos
        Relative /= np.linalg.norm(Relative)
        
        AngleCrit = arccos(np.dot(EdgeVec,Relative))
        
        
        return AngleCrit
    

    def GlobalAngle(self,Edge,Angle): # Unique angles always pointing inwards.
        
        if Edge == 'B':
            
            return Angle
        
        if Edge == 'R':
            
            GAngle = pi-self.Triangle.AnB + Angle #[pi-AnB,2pi-AnB]
            
            if GAngle >= pi:
                GAngle -= 2*pi
            
            return  GAngle
            
        if Edge == 'L':
            
            return -pi + self.Triangle.AnA + Angle #[-pi,AnA]
        
        
        
    def LocalAngle(self,Edge,Angle): # Unique angles always pointing inwards.
        
        if Edge == 'B':
            
            return Angle
        
        if Edge == 'R':
            
            NAngle = Angle - pi + self.Triangle.AnB  #[pi-AnB,2pi-AnB]
            
            return  NAngle
            
        if Edge == 'L':
            
            return pi - self.Triangle.AnA - Angle #[-pi,AnA]
        
        
        
        
    def SolvePoint(self,Point, EdgeNext, GlobAngle):
                
        v0x = cos(GlobAngle)
        v0y = sin(GlobAngle)
        
        Vertex = self.Triangle.Shape[EdgeNext]
        
        PtStart = Vertex[0]
        PtEnd = Vertex[1]
        
        v1 = PtEnd - PtStart
        
        Matrix = np.zeros([2,2])
        
        Matrix[0,0] = v0x
        Matrix[1,0] = v0y
        
        Matrix[:,1] = v1
        
        Target = PtStart - Point
        
        Answer = np.linalg.solve(Matrix,Target)
        
        PtNext = Point + Answer[0]*Matrix[:,0]
        
        return PtNext
        
        
        
    def GlobAngleReflector(self,Edge,Angle):
        
        if Edge == 'B':
            NAngle = - Angle
            
        elif Edge == 'L':
            NAngle = 2*self.Triangle.AnA - Angle
            
        elif Edge == 'R':
            NAngle = -2*self.Triangle.AnB - Angle
            
        if NAngle <= -pi:
            NAngle += 2*pi
            
        elif NAngle > pi:
            NAngle -= 2*pi
            
        return NAngle
            
        
    def Glance(self):
        print(f"FW TRGPO PYTHON VERSION - MARCH 2021")
        print(f"Initial c: {self.c}")
        print(f"Initial Th: {self.Th}")

    
    def Evolve(self, Log = True):
        
        Periodic = False
        
        self.Glance()
        
        c = self.c
        
        if Log:
            TLog = np.zeros([self.Iter,2])
            PtLog = np.zeros([self.Iter,2])
        
        Edge = self.CurrentEdge
        Pos = self.TranslateLocation()  
        
        Th = self.Th
        ThCrit = self.CriticalAngles()
                
        Angle = self.GlobalAngle(Edge, Th)
        
        for ii in range(self.Iter):
            
            print('\n\n\n\n')
            print(f'{ii}, {Th}')
        
            if Th == ThCrit or Th == 0:
                
                raise ValueError("Billiard Hit Vertex.")
                
            elif Th < ThCrit: # Reach Next Edge
            
                EdgeNext = self.Triangle.Edges[int(floor(c)) + 1]
            

            else: # Reach Previous Edge
            
                EdgeNext = self.Triangle.Edges[int(floor(c)) - 1]
                
            
            print(EdgeNext)
            
            PtNext = self.SolvePoint(Pos, EdgeNext,Angle)
            
            
            print(f'The next point, {PtNext}, should lie on {EdgeNext}')
            
            cNext = self.TranslateC(EdgeNext, PtNext)
            
            print(f'{Pos},{PtNext}')
            
            AngleNext = self.GlobAngleReflector(EdgeNext,Angle)
            
            print(f'Current Global Angle: {Angle}, Next Global Angle: {AngleNext}')
            
            ThNext = self.LocalAngle(EdgeNext, AngleNext)
            
            print(ThNext)
            
            if ThNext <= 0 and ThNext >= -pi:
                ThNext += pi
            
            elif ThNext > pi and ThNext <= 2*pi:
                
                ThNext -= pi
                
            elif ThNext <= pi and ThNext >= 0:
                
                print(f'Run {ii} Passed!')
                
            else:
                
                print(f'Something is Seriously Wrong at Iteration {ii}!')

                break
            
            TLog[ii,0] = c
            TLog[ii,1] = Th
            
            PtLog[ii,:] = Pos
            
            
            self.c = cNext
            self.Th = ThNext
            self.CurrentEdge = EdgeNext
            
            
            print(f'C = {self.c}')
            Pos = self.TranslateLocation()
            ThCrit = self.CriticalAngles()
            
            print(f'From {Pos}({Edge}), the critical angle is {ThCrit}')
            
            
            c = cNext
            Th = ThNext     
            Edge = self.CurrentEdge
            
            Angle = self.GlobalAngle(Edge, Th)
            
        if Log:
            return TLog,PtLog
            
        


M = Triangle(1,1.2)
x = Billiard(1.3146,pi-0.6, M)

TLog,PtLog = x.Evolve()

print(x.TranslateC(x.CurrentEdge,x.TranslateLocation()))


# Public Methods Cont'd


M.Plot()
plt.plot(PtLog[:,0],PtLog[:,1])
            
        