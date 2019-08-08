from pyoptsparse import Optimization, SNOPT, pyOpt_solution, NSGA2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from position_constraints import calculate_boundary
from windRoses import *
import os
from aep_calc import calcAEP
import constraints
from yy_calc_fatigue import *
import time

import sys
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
    global Omega_free
    global free_speed
    global Omega_close
    global close_speed
    global Omega_far
    global far_speed
    global Rhub
    global r
    global chord
    global theta
    global af
    global Rtip
    global B
    global rho
    global mu
    global precone
    global hubHt
    global nSector
    global pitch
    global yaw_deg
    global TI
    global scale


    turbineX = xdict['turbineX']*scale
    turbineY = xdict['turbineY']*scale

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

    AEP = calcAEP(turbineX,turbineY,windDirections,windSpeeds,windFrequencies,TI=TI)
    funcs['AEP'] = -AEP/1.E8


    separation_squared,boundary_distances = constraints.constraints_position(turbineX,turbineY,boundaryVertices,boundaryNormals)

    funcs['sep'] = (separation_squared-(126.4*2.)**2)/1.E5
    bounds = boundary_distances
    # funcs['sep'] = SpacingConstraint(turbineX, turbineY, rotorDiameter, minSpacing=2.0)/1.E5
    # bounds = arbitraryBoundary(turbineX, turbineY, boundaryVertices, boundaryNormals)/1.E3
    b = np.zeros(np.shape(bounds)[0])
    for i in range(len(b)):
        b[i] = min(bounds[i])
    funcs['bound'] = b

    damage = farm_damage(turbineX,turbineY,windDirections,windFrequencies,Omega_free,free_speed,
                                    Omega_close,close_speed,Omega_far,far_speed,Rhub,r,chord,theta,af,Rtip,
                                    B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=TI)

    funcs['damage'] = np.max(damage)
    fail = False

    return funcs, fail


def obj_func_damage(xdict):
    global windDirections
    global windSpeeds
    global windFrequencies
    global boundaryVertices
    global boundaryNormals
    global Omega_free
    global free_speed
    global Omega_close
    global close_speed
    global Omega_far
    global far_speed
    global Rhub
    global r
    global chord
    global theta
    global af
    global Rtip
    global B
    global rho
    global mu
    global precone
    global hubHt
    global nSector
    global pitch
    global yaw_deg
    global TI
    global scale


    turbineX = xdict['turbineX']*scale
    turbineY = xdict['turbineY']*scale

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

    AEP = calcAEP(turbineX,turbineY,windDirections,windSpeeds,windFrequencies,TI=TI)
    funcs['AEP'] = -AEP/1.E7

    separation_squared,boundary_distances = constraints.constraints_position(turbineX,turbineY,boundaryVertices,boundaryNormals)


    funcs['sep'] = (separation_squared-(126.4*2.)**2)/1.E5
    bounds = boundary_distances
    # funcs['sep'] = SpacingConstraint(turbineX, turbineY, rotorDiameter, minSpacing=2.0)/1.E5
    # bounds = arbitraryBoundary(turbineX, turbineY, boundaryVertices, boundaryNormals)/1.E3
    b = np.zeros(np.shape(bounds)[0])
    for i in range(len(b)):
        b[i] = min(bounds[i])
    funcs['bound'] = b

    damage = farm_damage(turbineX,turbineY,windDirections,windFrequencies,Omega_free,free_speed,
                                    Omega_close,close_speed,Omega_far,far_speed,Rhub,r,chord,theta,af,Rtip,
                                    B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg,TI=TI)
    funcs['damage'] = damage
    fail = False

    return funcs, fail


## Main
if __name__ == "__main__":

    global windDirections
    global windSpeeds
    global windFrequencies
    global boundaryVertices
    global boundaryNormals
    global Omega_free
    global free_speed
    global Omega_close
    global close_speed
    global Omega_far
    global far_speed
    global Rhub
    global r
    global chord
    global theta
    global af
    global Rtip
    global B
    global rho
    global mu
    global precone
    global hubHt
    global nSector
    global pitch
    global yaw_deg
    global TI
    global scale

    print 'running setup'

    filename_free = '/home/flowlab/PJ/waked-loads/BYU/BYU/C680_W8_T11.0_P0.0_m2D_L0/Model.out'
    filename_close = '/home/flowlab/PJ/waked-loads/BYU/BYU/C653_W8_T11.0_P0.0_4D_L0/Model.out'
    filename_far = '/home/flowlab/PJ/waked-loads/BYU/BYU/C671_W8_T11.0_P0.0_10D_L0/Model.out'

    TI = 0.11
    Omega_free,free_speed,Omega_close,close_speed,Omega_far,far_speed = find_omega(filename_free,filename_close,filename_far,TI=TI)
    Rhub,r,chord,theta,af,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg = setup_airfoil()

    nTurbs = 10

    damage_lim = 1.21
    folder = 'yy_results/10turbs_2dirs_cons1.21'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # windDirections, windFrequencies, windSpeeds = northIslandRose(30)

    # windDirections = np.array([270.])
    # windSpeeds = np.array([8.])
    # windFrequencies = np.array([1.])

    windDirections = np.array([0.,270.])
    windSpeeds = np.array([8.,8.])
    windFrequencies = np.array([0.5,0.5])

    rotor_diameter = 126.4
    spacing = 3.
    side_length = (np.sqrt(nTurbs)-1.)*rotor_diameter*spacing
    a = side_length**2
    circle_radius = np.sqrt(a/np.pi)

    boundary = 'circle'

    if boundary == 'circle':
        nBounds = 50
        theta_circ = np.linspace(0.,360.-360./float(nBounds),nBounds)
        xBounds = np.zeros(nBounds)
        yBounds = np.zeros(nBounds)
        for i in range(nBounds):
            xBounds[i] = circle_radius*np.cos(np.deg2rad(theta_circ[i]))
            yBounds[i] = circle_radius*np.sin(np.deg2rad(theta_circ[i]))

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
        locations = np.loadtxt('/Users/ningrsrch/Dropbox/Projects/waked-loads/layout_amalia.txt')
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



    num = 1000

    print 'running optimization'

    scale = 10.

    for k in range(num):
        turbineX,turbineY = random_start(nTurbs,rotor_diameter,xmin,xmax,ymin,ymax)
        turbineX = turbineX/scale
        turbineY = turbineY/scale

        """Optimization"""
        optProb = Optimization('Wind_Farm_AEP', obj_func_damage)
        optProb.addObj('AEP')
        # optProb.addObj('damage')

        optProb.addVarGroup('turbineX', nTurbs, type='c', lower=min(xBounds), upper=max(xBounds), value=turbineX)
        optProb.addVarGroup('turbineY', nTurbs, type='c', lower=min(yBounds), upper=max(yBounds), value=turbineY)

        num_cons_sep = (nTurbs-1)*nTurbs/2
        optProb.addConGroup('sep', num_cons_sep, lower=0., upper=None)
        optProb.addConGroup('bound', nTurbs, lower=0., upper=None)
        optProb.addConGroup('damage', nTurbs, lower=None, upper=damage_lim)

        opt = SNOPT()
        opt.setOption('Scale option',0)
        opt.setOption('Iterations limit',1000000)

        opt.setOption('Summary file','summary.out')
        opt.setOption('Major optimality tolerance',1.e-4)
        opt.setOption('Major feasibility tolerance',1.e-6)

        res = opt(optProb)

        x = res.xStar['turbineX']
        y = res.xStar['turbineY']

        input = {'turbineX':x,'turbineY':y}
        funcs,_ = obj_func_damage(input)

        separation = min(funcs['sep'])
        boundary = min(funcs['bound'])
        AEP = -funcs['AEP']
        damage = funcs['damage']

        tol = 1.E-4
        if separation > -tol and boundary > -tol and max(damage) < damage_lim+tol:
        # if separation > -tol and boundary > -tol:

            file = open('%s/AEP.txt'%folder, 'a')
            file.write('%s'%(AEP) + '\n')
            file.close()

            file = open('%s/damage.txt'%folder, 'a')
            for turb in range(nTurbs):
                file.write('%s'%damage[turb] + ' ')
            file.write('\n')
            file.close()

            file = open('%s/turbineX.txt'%folder, 'a')
            for turb in range(nTurbs):
                file.write('%s'%(x[turb]*scale) + ' ')
            file.write('\n')
            file.close()

            file = open('%s/turbineY.txt'%folder, 'a')
            for turb in range(nTurbs):
                file.write('%s'%(y[turb]*scale) + ' ')
            file.write('\n')
            file.close()
