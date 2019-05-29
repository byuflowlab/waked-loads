from pyoptsparse import Optimization, SNOPT, pyOpt_solution, NSGA2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from position_constraints import *
from windRoses import *
# from grid_param import *
from aep_calc import *
import os
import sys
sys.dont_write_bytecode = True

def obj_func(xdict):
    global rotorDiameter
    global turbineZ
    global windDirections
    global windSpeeds
    global windFrequencies
    global shearExp
    global minSpacing
    global nTurbs
    global boundaryVertices
    global boundaryNormals

    turbineX = xdict['x']
    turbineY = xdict['y']

    funcs = {}
    AEP, overlap = calcAEP(turbineX,turbineY,turbineZ,rotorDiameter,windDirections,windSpeeds,windFrequencies,shearExp,
                wakemodel='floris')
    funcs['obj'] = -AEP/1.E5
    funcs['overlap'] = overlap

    funcs['sep'] = SpacingConstraint(turbineX, turbineY, rotorDiameter, minSpacing=minSpacing)/1.E5
    bounds = arbitraryBoundary(turbineX, turbineY, boundaryVertices, boundaryNormals)/1.E3
    b = np.zeros(np.shape(bounds)[0])
    for i in range(len(b)):
        b[i] = min(bounds[i])
    funcs['bound'] = b

    fail = False

    return funcs, fail


## Main
if __name__ == "__main__":

    global rotorDiameter
    global turbineZ
    global windDirections
    global windSpeeds
    global windFrequencies
    global shearExp
    global minSpacing
    global nTurbs
    global boundaryVertices
    global boundaryNormals

    wakemodel = "floris"

    nTurbs = 20

    rose = 'northIsland'
    # windDirections, windFrequencies, windSpeeds = northIslandRose(30)
    # wind_angle = windDirections[np.argmax(windFrequencies)]
    #
    # windDirections, windFrequencies, windSpeeds = northIslandRose(30,nSpeeds=8)
    # windDirections -= wind_angle
    windDirections = np.array([0.])
    windSpeeds = np.array([10.])
    windFrequencies = np.array([1.])

    turbineZ = np.ones(nTurbs)*100.
    rotorDiameter = np.ones(nTurbs)*130.

    shearExp = 0.15
    minSpacing = 2.0

    spacing = 4.
    side_length = (np.sqrt(nTurbs)-1.)*rotorDiameter[0]*spacing
    a = side_length**2
    circle_radius = np.sqrt(a/np.pi)

    """amalia boundary"""
    locations = np.loadtxt('layout_amalia.txt')
    xBounds = locations[:, 0]
    yBounds = locations[:, 1]
    xBounds = xBounds - min(xBounds) - (max(xBounds)-min(xBounds))/2.
    yBounds = yBounds - min(yBounds) - (max(yBounds)-min(yBounds))/2.
    locations[:, 0] = xBounds
    locations[:, 1] = yBounds
    boundaryVertices, boundaryNormals = calculate_boundary(locations)
    xBounds = boundaryVertices[:, 0]
    yBounds = boundaryVertices[:, 1]

    nBounds = len(xBounds)
    points = np.zeros((nBounds,2))
    points[:, 0] = xBounds
    points[:, 1] = yBounds

    hull = sp.spatial.ConvexHull(points)
    area = hull.volume
    area_ratio = area/(np.pi*circle_radius**2)
    xBounds = xBounds/np.sqrt(area_ratio)
    yBounds = yBounds/np.sqrt(area_ratio)
    points[:, 0] = xBounds
    points[:, 1] = yBounds

    boundaryVertices, boundaryNormals = calculate_boundary(points)

    max_overlap = np.array([15.,14.5,14.,13.5,13.,12.5,12.,11.5,11.,10.5,10.,9.5,9.,8.5,8.,7.5,7.,6.5,6.,5.5,5.,4.5,4.,3.5,3.,2.5,2.,1.5,1.,0.5])
    num = len(max_overlap)

    aep_array = np.zeros(num)
    overlap_array = np.zeros(num)

    for i in range(num):
        print max_overlap[i]
        maxAEP = 0.
        maxAEP_overlap = 0.
        for j in range(15):
            x = np.random.rand(nTurbs)*(max(xBounds)-min(xBounds))+min(xBounds)
            y = np.random.rand(nTurbs)*(max(yBounds)-min(yBounds))+min(yBounds)

            """Optimization"""
            optProb = Optimization('Wind_Farm_AEP', obj_func)
            optProb.addObj('obj')

            optProb.addVarGroup('x', nTurbs, type='c', lower=min(xBounds), upper=max(xBounds), value=x)
            optProb.addVarGroup('y', nTurbs, type='c', lower=min(yBounds), upper=max(yBounds), value=y)

            num_cons_sep = (nTurbs-1)*nTurbs/2
            optProb.addConGroup('sep', num_cons_sep, lower=0., upper=None)
            optProb.addConGroup('bound', nTurbs, lower=0., upper=None)
            optProb.addCon('overlap', lower=None, upper=max_overlap[i])


            opt = SNOPT()
            opt.setOption('Scale option',0)
            opt.setOption('Iterations limit',1000000)

            opt.setOption('Summary file','current_summary_square_gaus.out')
            opt.setOption('Major optimality tolerance',1.e-5)
            opt.setOption('Major feasibility tolerance',1.e-6)

            res = opt(optProb)

            x = res.xStar['x']
            y = res.xStar['y']

            input = {'x':x,'y':y}
            funcs,_ = obj_func(input)

            separation = min(funcs['sep'])
            boundary = min(funcs['bound'])

            if separation > -1.E-4 and boundary > -1.E-4 and -funcs['obj'] > maxAEP:
                maxAEP = -funcs['obj']
                maxAEP_overlap = funcs['overlap']
                print maxAEP
                print maxAEP_overlap



        aep_array[i] = maxAEP
        overlap_array[i] = maxAEP_overlap
        print 'AEP: ', repr(aep_array)
        print 'overlap: ', repr(overlap_array)
