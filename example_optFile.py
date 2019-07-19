from pyoptsparse import Optimization, SNOPT, pyOpt_solution, NSGA2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from position_constraints import *
from windRoses import *
import os
import sys
import fast_calc_aep
import constraints
from calc_fatigue_NAWEA import *
import time

sys.dont_write_bytecode = True


def random_start(N,D,xmin,xmax,ymin,ymax):
    x = np.zeros(N)
    y = np.zeros(N)
    i = 0
    while i < N:
        good = True
        xtemp = float(np.random.rand(1))*(xmax-xmin)+xmin
        ytemp = float(np.random.rand(1))*(ymax-ymin)+ymin
        for j in range(i):
            dist = np.sqrt((x[j]-xtemp)**2+(y[j]-ytemp)**2)
            if dist < D:
                good = False
        if good == True:
            x[i] = xtemp
            y[i] = ytemp
            i += 1
    return x,y


def obj_func(xdict):
    global windDirections
    global windSpeeds
    global windFrequencies
    global boundaryVertices
    global boundaryNormals
    global damage_free
    global damage_close
    global damage_far


    turbineX = xdict['turbineX']
    turbineY = xdict['turbineY']

    """turbine definition"""
    turbineZ = np.ones_like(turbineX)*90.
    rotorDiameter = np.ones_like(turbineX)*126.4
    shearExp = 0.
    wakemodel = 2
    relaxationFactor = 1.
    rated_ws = 11.4
    rated_power = 5000.
    cut_in_speed = 3.
    cut_out_speed = 25.
    zref = 90.
    z0 = 0.


    funcs = {}

    AEP = fast_calc_aep.calcaep(turbineX, turbineY, turbineZ, rotorDiameter, windDirections,
                windSpeeds, windFrequencies, shearExp, wakemodel, relaxationFactor, rated_ws, rated_power,
                cut_in_speed, cut_out_speed, zref, z0)


    funcs['AEP'] = -AEP/1.E6


    separation_squared,boundary_distances = constraints.constraints_position(turbineX,turbineY,boundaryVertices,boundaryNormals)

    funcs['sep'] = (separation_squared-(126.4*2.)**2)/1.E5
    bounds = boundary_distances
    # funcs['sep'] = SpacingConstraint(turbineX, turbineY, rotorDiameter, minSpacing=2.0)/1.E5
    # bounds = arbitraryBoundary(turbineX, turbineY, boundaryVertices, boundaryNormals)/1.E3
    b = np.zeros(np.shape(bounds)[0])
    for i in range(len(b)):
        b[i] = min(bounds[i])
    funcs['bound'] = b

    funcs['damage'] = farm_damage(turbineX,turbineY,windDirections,windFrequencies,damage_free,damage_close,damage_far)
    fail = False

    return funcs, fail


def final_call(xdict):
    global windDirections
    global windSpeeds
    global windFrequencies
    global boundaryVertices
    global boundaryNormals
    global damage_free
    global damage_close
    global damage_far


    turbineX = xdict['turbineX']
    turbineY = xdict['turbineY']

    """turbine definition"""
    turbineZ = np.ones_like(turbineX)*90.
    rotorDiameter = np.ones_like(turbineX)*126.4
    shearExp = 0.
    wakemodel = 2
    relaxationFactor = 1.
    rated_ws = 11.4
    rated_power = 5000.
    cut_in_speed = 3.
    cut_out_speed = 25.
    zref = 90.
    z0 = 0.


    funcs = {}

    AEP = fast_calc_aep.calcaep(turbineX, turbineY, turbineZ, rotorDiameter, windDirections,
                windSpeeds, windFrequencies, shearExp, wakemodel, relaxationFactor, rated_ws, rated_power,
                cut_in_speed, cut_out_speed, zref, z0)


    funcs['AEP'] = -AEP/1.E6


    separation_squared,boundary_distances = constraints.constraints_position(turbineX,turbineY,boundaryVertices,boundaryNormals)

    funcs['sep'] = (separation_squared-(126.4*2.)**2)/1.E5
    bounds = boundary_distances
    # funcs['sep'] = SpacingConstraint(turbineX, turbineY, rotorDiameter, minSpacing=2.0)/1.E5
    # bounds = arbitraryBoundary(turbineX, turbineY, boundaryVertices, boundaryNormals)/1.E3
    b = np.zeros(np.shape(bounds)[0])
    for i in range(len(b)):
        b[i] = min(bounds[i])
    funcs['bound'] = b

    funcs['damage'] = farm_damage(turbineX,turbineY,windDirections,windFrequencies,damage_free,damage_close,damage_far)
    fail = False

    return funcs, fail


## Main
if __name__ == "__main__":

    global windDirections
    global windSpeeds
    global windFrequencies
    global boundaryVertices
    global boundaryNormals
    global damage_free
    global damage_close
    global damage_far



    filename_free = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
    filename_close = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
    filename_far = '/Users/ningrsrch/Dropbox/Projects/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'

    print 'setup fatigue model'
    s = time.time()
    damage_free,damage_close,damage_far = extract_damage(filename_free,filename_close,filename_far)
    print 'setup time: ', time.time()-s

    print 'running optimization'

    nTurbs = 10

    # windDirections, windFrequencies, windSpeeds = northIslandRose(30)

    windDirections = np.array([270.,180.])
    windSpeeds = np.array([8.,8.])
    windFrequencies = np.array([0.5,0.5])

    windDirections = np.array([270.])
    windSpeeds = np.array([8.])
    windFrequencies = np.array([1.])

    rotor_diameter = 126.4
    spacing = 3.
    side_length = (np.sqrt(nTurbs)-1.)*rotor_diameter*spacing
    a = side_length**2
    circle_radius = np.sqrt(a/np.pi)

    boundary = 'circle'

    if boundary == 'circle':
        nBounds = 50
        theta = np.linspace(0.,360.-360./float(nBounds),nBounds)
        xBounds = np.zeros(nBounds)
        yBounds = np.zeros(nBounds)
        for i in range(nBounds):
            xBounds[i] = circle_radius*np.cos(np.deg2rad(theta[i]))
            yBounds[i] = circle_radius*np.sin(np.deg2rad(theta[i]))

        locations = np.zeros((nBounds,2))
        locations[:, 0] = xBounds
        locations[:, 1] = yBounds
        boundaryVertices, boundaryNormals = calculate_boundary(locations)

    elif boundary == 'square':
        nBounds = 4
        x = np.array([-side_length/2.,side_length/2.,side_length/2.,-side_length/2.])
        y = np.array([-side_length/2.,-side_length/2.,side_length/2.,side_length/2.])
        xBounds = x*np.cos(np.deg2rad(30.)) - y*np.sin(np.deg2rad(30.))
        yBounds = x*np.sin(np.deg2rad(30.)) + y*np.cos(np.deg2rad(30.))

        locations = np.zeros((nBounds,2))
        locations[:, 0] = xBounds
        locations[:, 1] = yBounds
        boundaryVertices, boundaryNormals = calculate_boundary(locations)
    elif boundary == 'amalia':
        locations = np.loadtxt('/home/flowlab/PJ/reduction/layout_amalia.txt')
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
        area_ratio = area/a
        xBounds = xBounds/np.sqrt(area_ratio)
        yBounds = yBounds/np.sqrt(area_ratio)

        locations = np.zeros((len(xBounds),2))
        locations[:, 0] = xBounds
        locations[:, 1] = yBounds
        boundaryVertices, boundaryNormals = calculate_boundary(locations)

    if boundary == 'circle':
        xmin = -circle_radius
        xmax = circle_radius
        ymin = -circle_radius
        ymax = circle_radius
    elif boundary == 'square':
        xmax = side_length/2.
        xmin = -side_length/2.
        ymax = side_length/2.
        ymin = -side_length/2.
    elif boundary =='amalia':
        xmax = max(xBounds)
        xmin = min(xBounds)
        ymax = max(yBounds)
        ymin = min(yBounds)

    turbineX,turbineY = random_start(nTurbs,126.4,xmin,xmax,ymin,ymax)

    """Optimization"""
    optProb = Optimization('Wind_Farm_AEP', obj_func)
    optProb.addObj('AEP')

    optProb.addVarGroup('turbineX', nTurbs, type='c', lower=min(xBounds), upper=max(xBounds), value=turbineX)
    optProb.addVarGroup('turbineY', nTurbs, type='c', lower=min(yBounds), upper=max(yBounds), value=turbineY)

    num_cons_sep = (nTurbs-1)*nTurbs/2
    optProb.addConGroup('sep', num_cons_sep, lower=0., upper=None)
    optProb.addConGroup('bound', nTurbs, lower=0., upper=None)
    # optProb.addConGroup('damage', nTurbs, lower=None, upper=0.8)

    opt = SNOPT()
    opt.setOption('Scale option',0)
    opt.setOption('Iterations limit',1000000)

    opt.setOption('Summary file','summary.out')
    opt.setOption('Major optimality tolerance',1.e-5)
    opt.setOption('Major feasibility tolerance',1.e-6)

    res = opt(optProb)

    x = res.xStar['turbineX']
    y = res.xStar['turbineY']

    print 'x: ', x

    input = {'turbineX':x,'turbineY':y}
    funcs,_ = final_call(input)

    separation = min(funcs['sep'])
    boundary = min(funcs['bound'])
    AEP = -funcs['AEP']

    print 'separation: ', separation
    print 'boundary: ', boundary
    print 'AEP: ', AEP
    print 'damage: ', funcs['damage']

    # int = np.argsort(x)
    # sortedX = np.zeros(len(x))
    # sortedY = np.zeros(len(y))
    # for k in range(len(x)):
    #     sortedX[k] = x[int[k]]
    #     sortedY[k] = y[int[k]]

    for i in range(len(x)):
        circ = plt.Circle((x[i],y[i]), 126.4/2.,facecolor='blue',edgecolor='blue',alpha=0.5)
        plt.gca().add_patch(circ)
        plt.text(x[i],y[i],'%s'%(i+1))

    plt.xlim(-3000.,3000.)
    plt.axis('equal')

    bx = boundaryVertices[:,0]
    by = boundaryVertices[:,1]
    bx = np.append(bx,bx[0])
    by = np.append(by,by[0])
    plt.plot(bx,by,'--k')
    plt.show()
