#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Interactive Triangular Billiard Music Program!
"""


from TrgPO import Triangle, Billiard

import numpy as np

import pylab

from numpy import pi, cos, sin, tan, floor, ceil, exp, arccos, arcsin, arctan

import numexpr as ne

import pygame
from pygame.locals import *

import sys

import matplotlib.pyplot as plt # Temporary solution
import matplotlib
matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg


M = Triangle(pi/4,pi/4)

x = Billiard(1.5,1, M, Iter = 50)

fig, ax = x.Triangle.Plot()


canvas = agg.FigureCanvasAgg(fig)
canvas.draw()
renderer = canvas.get_renderer()
raw_data = renderer.tostring_rgb()



pygame.init()

window = pygame.display.set_mode((600, 400), DOUBLEBUF)
screen = pygame.display.get_surface()

size = canvas.get_width_height()

surf = pygame.image.fromstring(raw_data, size, "RGB")
screen.blit(surf, (0,0))
pygame.display.flip()

pygame.display.set_caption("FWPhys Triangular Billiard App")

crashed = False
while not crashed:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			crashed = True





    