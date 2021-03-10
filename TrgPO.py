#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 19:51:49 2021

@author: ywan598
"""

import numpy as np
import matplotlib.pyplot as plt # Temporary solution

from numpy import pi, cos, sin, tan, floor, ceil, exp, arccos, arcsin, arctan




def EAS(Angle):
    
    Angle = arctan(tan(Angle))
    
    if Angle <= 0:
        Angle += pi
        
    return Angle
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
        
        self.Edges = ["B","R","L","B","R","L"]
        
        
            
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
        
        ax.plot(Vertex[:,0],Vertex[:,1])
        ax.set_aspect('equal')
        ax.grid()

        return fig, ax
################ BUTTER ################


class Billiard:
    
    def __init__(self,c0,Th0, Triangle, Iter = 200):
        
            
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
        
        
        Relative = PtOff - Pos
        Relative /= np.linalg.norm(Relative)
        
        AngleCrit = arccos(np.dot(EdgeVec,Relative))
        
        
        return AngleCrit
    

    def GlobalAngle(self,Edge,Angle): # Unique angles between 0 and pi.
        
        if Edge == 'B':
            
            GAngle = Angle
        
        if Edge == 'R':
            
            GAngle = pi-self.Triangle.AnB + Angle #[pi-AnB,2pi-AnB]

            
        if Edge == 'L':
            
            GAngle = -pi + self.Triangle.AnA + Angle #[-pi,AnA]
        
        
        return EAS(GAngle)

    def LocalAngle(self,Edge,Angle): # Unique angles always pointing inwards.
        
        if Edge == 'B':
            
            LAngle = Angle
        
        if Edge == 'R':
            
            LAngle = Angle - pi + self.Triangle.AnB  #[pi-AnB,2pi-AnB]
            
            
        if Edge == 'L':
            
            LAngle = pi - self.Triangle.AnA - Angle #[-pi,AnA]
        
        return EAS(LAngle)
        
        
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
            NAngle = pi - Angle
            
        elif Edge == 'L':
            NAngle = -2*self.Triangle.AnA + Angle
        
        elif Edge == 'R':
            NAngle = -2*self.Triangle.AnB - Angle

            
        return EAS(NAngle)
            
        
    def Glance(self):
        print(f"FW TRGPO PYTHON VERSION - MARCH 2021")
        print(f"Initial c: {self.c}")
        print(f"Initial Th: {self.Th}")

    
    def Evolve(self, Log = True, NumTol = 1e-6):
        
        Periodic = False
        
        self.Glance()
        
        c = self.c


        
        
        if Log:
            TLog = np.zeros([self.Iter,2])
            PtLog = np.zeros([self.Iter,2])

        
        Edge = self.CurrentEdge
        Word = str(Edge)
        
        
        Pos = self.TranslateLocation()  
        
        Th = self.Th
        ThCrit = self.CriticalAngles()
                
        Angle = self.GlobalAngle(Edge, Th)
        
        cInt = c
        ThInt = Th
        
        if Th == ThCrit or Th == 0:
                
            raise ValueError("Billiard Hit Vertex.")
                
        elif Th < ThCrit: # Reach Next Edge
            
            EdgeNext = self.Triangle.Edges[int(floor(c)) + 1]
            

        else: # Reach Previous Edge
            
            EdgeNext = self.Triangle.Edges[int(floor(c)) - 1]
        
        for ii in range(self.Iter):
            
            print(f'{ii}',end = '\r')
            
            PtNext = self.SolvePoint(Pos, EdgeNext,Angle)
            

            
            cNext = self.TranslateC(EdgeNext, PtNext)
            
            AngleNext = self.GlobAngleReflector(EdgeNext,Angle)

            
            ThNext = self.LocalAngle(EdgeNext, AngleNext)
            

            
            if ThNext <= 0 and ThNext >= -pi:
                ThNext += pi
            
            elif ThNext > pi and ThNext <= 2*pi:
                
                ThNext -= pi
                
            
            TLog[ii,0] = c
            TLog[ii,1] = Th
            
            PtLog[ii,:] = Pos
            
            
            self.c = cNext
            self.Th = ThNext
            self.CurrentEdge = EdgeNext

            Pos = self.TranslateLocation()
            ThCrit = self.CriticalAngles()

      
            c = cNext
            Th = ThNext     
            Edge = self.CurrentEdge
            Word += Edge
            Angle = self.GlobalAngle(Edge, Th)
            
                
            if Th == ThCrit or Th == 0:
                    
                raise ValueError("Billiard Hit Vertex.")
                    
            elif Th < ThCrit: # Reach Next Edge
                
                EdgeNext = self.Triangle.Edges[int(floor(c)) + 1]
                    
            else: # Reach Previous Edge
                
                EdgeNext = self.Triangle.Edges[int(floor(c)) - 1]
                                
            if Periodic:
                
                print(f'We found Periodic Orbit Within Numerical Limits. \nPeriod is {ii}.')
                
                return TLog[0:ii,:], PtLog[0:ii,::], Periodic
        
            Deviation = np.sqrt((c-cInt)**2 + (Th-ThInt)**2)
            
            if Deviation <= NumTol:
                Periodic = True
                print(f'With period word {Word}')
                
                self.Word = Word
                self.Periodic = True
                self.Period = ii + 1

        print(f'Wihin {self.Iter} attempts, we did not find a Periodic Orbit.')
        
        self.Word = ''
        self.Periodic = False
        self.Period = np.nan
        
        if Log:
            return TLog,PtLog, Periodic




######################################## Let's Roll!

        