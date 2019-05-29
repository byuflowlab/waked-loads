import numpy as np
import _floris
import fast_aep_calc
from wake_models import *
import sys
sys.dont_write_bytecode = True


def WindFrame(wind_direction, turbineX, turbineY):
    """ Calculates the locations of each turbine in the wind direction reference frame """
    nTurbines = len(turbineX)
    windDirectionDeg = wind_direction
    # adjust directions
    windDirectionDeg = 270. - windDirectionDeg
    if windDirectionDeg < 0.:
        windDirectionDeg += 360.
    windDirectionRad = np.pi*windDirectionDeg/180.0    # inflow wind direction in radians

    # convert to downwind(x)-crosswind(y) coordinates
    turbineXw = turbineX*np.cos(-windDirectionRad)-turbineY*np.sin(-windDirectionRad)
    turbineYw = turbineX*np.sin(-windDirectionRad)+turbineY*np.cos(-windDirectionRad)

    # import matplotlib.pyplot as plt
    # plt.plot(turbineXw,turbineYw,'o')
    # plt.xlim(-1000.,3000.)
    # plt.ylim(-1000.,3000.)
    # plt.show()

    return turbineXw, turbineYw


def PowWind(Uref, turbineZ, shearExp, zref=99999., z0=0.):
    """
    wind shear power law
    """
    if zref == 99999.:
        zref = turbineZ[0]
    nTurbines = len(turbineZ)

    turbineSpeeds = np.zeros(nTurbines)

    for turbine_id in range(nTurbines):
        turbineSpeeds[turbine_id]= Uref*((turbineZ[turbine_id]-z0)/(zref-z0))**shearExp

    return turbineSpeeds


def WindDirectionPower(wtVelocity,rated_ws=False,rated_power=False,cut_in_speed=False,cut_out_speed=False):
    """calculate power from a given wind direction"""
    nTurbines = len(wtVelocity)

    if rated_ws == False:
        rated_ws = 11.4
    if rated_power == False:
        rated_power = 5.
    if cut_in_speed == False:
        cut_in_speed = 4.
    if cut_out_speed == False:
        cut_out_speed = 25.

    wtPower = np.zeros(nTurbines)
    buffer = 0.1
    for turb in range(nTurbines):
        # If we're below cut-in
        if wtVelocity[turb] < (cut_in_speed-buffer):
            wtPower[turb] = 0.
        # If we're at the spline of cut-in
        if ((wtVelocity[turb] > (cut_in_speed-buffer)) and (wtVelocity[turb] < (cut_in_speed+buffer))):
            x0 = cut_in_speed-buffer
            x1 = cut_in_speed+buffer
            y0 = 0.
            y1 = rated_power*((cut_in_speed+buffer)/rated_ws)**3
            dy0 = 0.
            dy1 = 3.*rated_power*(cut_in_speed+buffer)**2/(rated_ws**3)
            wtPower[turb] = _floris.hermite_spline(wtVelocity[turb], x0, x1, y0, dy0, y1, dy1)
        # If we're between cut-in and rated
        if ((wtVelocity[turb] > (cut_in_speed+buffer)) and (wtVelocity[turb] < (rated_ws-buffer))):
            wtPower[turb] = rated_power*(wtVelocity[turb]/rated_ws)**3
        # If we're at the spline of rated
        if ((wtVelocity[turb] > (rated_ws-buffer)) and (wtVelocity[turb] < (rated_ws+buffer))):
            x0 = rated_ws-buffer
            x1 = rated_ws+buffer
            y0 = rated_power*((rated_ws-buffer)/rated_ws)**3
            y1 = rated_power
            dy0 = 3.*rated_power*(rated_ws-buffer)**2/(rated_ws**3)
            dy1 = 0.
            wtPower[turb] = _floris.hermite_spline(wtVelocity[turb], x0, x1, y0, dy0, y1, dy1)
        # If we're between rated and cut-out
        if ((wtVelocity[turb] > (rated_ws+buffer)) and (wtVelocity[turb] < (cut_out_speed-buffer))):
            wtPower[turb] = rated_power
        # If we're at the spline of cut-out
        if ((wtVelocity[turb] > (cut_out_speed-buffer)) and (wtVelocity[turb] < (cut_out_speed+buffer))):
            x0 = cut_out_speed-buffer
            x1 = cut_out_speed+buffer
            y0 = rated_power
            y1 = 0.
            dy0 = 0.
            dy1 = 0.
            wtPower[turb] = _floris.hermite_spline(wtVelocity[turb], x0, x1, y0, dy0, y1, dy1)
        # If we're above cut-out
        if wtVelocity[turb] > (cut_out_speed+buffer):
            wtPower[turb] = 0.

    # calculate total power for this direction
    dir_power = np.sum(wtPower)

    # pass out results
    return wtPower, dir_power


def calcAEP(turbineX,turbineY,turbineZ,rotorDiameter,windDirections,windSpeeds,windFrequencies,shearExp,
            wakemodel='jensen',relaxationFactor=1.0):

    nDirections = len(windDirections)

    dir_powers = np.zeros(nDirections)
    dir_overlap = np.zeros(nDirections)
    for i in range(nDirections):
        turbineXw, turbineYw = WindFrame(windDirections[i], turbineX,turbineY)
        Vinf = PowWind(windSpeeds[i], turbineZ, shearExp)

        if wakemodel=='jensen':
            wtVelocity = Jensen(turbineXw, turbineYw, rotorDiameter[0], Vinf, relaxationFactor=relaxationFactor)
        elif wakemodel=='floris':
            wtVelocity, wakeOverlapTRel = Floris(turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf)
        elif wakemodel=='gaussian':
            wtVelocity = Gaussian(turbineXw, turbineYw, rotorDiameter[0], Vinf)

        _,dir_powers[i] = WindDirectionPower(wtVelocity)
        if wakemodel=='floris':
            dir_overlap[i] = np.sum(wakeOverlapTRel)
    AEP = np.sum(dir_powers*windFrequencies*24.*365.)

    # print 'AEP slow: ', AEP
    if wakemodel=='floris':
        return AEP, np.sum(dir_overlap*windFrequencies)
    else:
        return AEP

def fast_calc_AEP(turbineX, turbineY, turbineZ, rotorDiameter, windDirections,
            windSpeeds, windFrequencies, wakemodel='jensen', rated_ws=9.8, rated_power=3.35,
            cut_in_speed=4., cut_out_speed=25., shearExp=0.15, relaxationFactor=1.0, zref=99999., z0=0.):

    if zref == 99999.:
        zref = turbineZ[0]

    if wakemodel == 'jensen':
        wakemodel_num = 1
    elif wakemodel == 'gaussian':
        wakemodel_num = 2
    elif wakemodel == 'floris':
        wakemodel_num = 3

    # print 'turbineX: ', turbineX
    # print 'turbineY: ', turbineY
    # print 'turbineZ: ', turbineZ
    # print 'rotorDiameter: ', rotorDiameter
    # print 'windDirections: ', windDirections
    # print 'windSpeeds: ', windSpeeds
    # print 'windFrequencies: ', windFrequencies
    # print 'shearExp: ', shearExp
    # print 'wakemodel_num: ', wakemodel_num
    # print 'relaxationFactor: ', relaxationFactor
    # print 'rated_ws: ', rated_ws
    # print 'rated_power: ', rated_power
    # print 'cut_in_speed: ', cut_in_speed
    # print 'cut_out_speed: ', cut_out_speed
    # print 'zref: ', zref
    # print 'z0: ', z0

    AEP = fast_aep_calc.calcaep(turbineX, turbineY, turbineZ, rotorDiameter, windDirections,
                windSpeeds, windFrequencies, shearExp, wakemodel_num, relaxationFactor, rated_ws, rated_power,
                cut_in_speed, cut_out_speed, zref, z0)

    # print 'AEP fast: ', AEP
    return AEP

if __name__=="__main__":
    # num = 10000
    # vel = np.linspace(0.,30.,num)
    # pow = np.zeros(num)
    # for i in range(num):
    #     wtVelocity = np.array([vel[i]])
    #     # print WindDirectionPower(wtVelocity)
    #     _,pow[i] = WindDirectionPower(wtVelocity)
    #
    # import matplotlib.pyplot as plt
    # plt.plot(vel,pow)
    # plt.show()

    # from windRoses import *
    # from grid_param import *
    # import time
    #
    # windDirections, windFrequencies, windSpeeds = victorvilleRose(30)
    # wind_angle = windDirections[np.argmax(windFrequencies)]
    #
    # nBounds = 20
    # circle_radius = 1370.
    # xBounds = np.zeros(nBounds)
    # yBounds = np.zeros(nBounds)
    # theta = np.linspace(0.,2.*np.pi-2.*np.pi/float(nBounds),nBounds)
    # for i in range(nBounds):
    #     xBounds[i] = circle_radius*np.cos(theta[i])
    #     yBounds[i] = circle_radius*np.sin(theta[i])
    # locations = np.zeros((nBounds,2))
    #
    # x = np.zeros_like(xBounds)
    # x[:] = xBounds[:]
    # y = np.zeros_like(yBounds)
    # y[:] = yBounds[:]
    # xBounds = x*np.cos(np.deg2rad(wind_angle)) - y*np.sin(np.deg2rad(wind_angle))
    # yBounds = x*np.sin(np.deg2rad(wind_angle)) + y*np.cos(np.deg2rad(wind_angle))
    #
    # locations[:, 0] = xBounds
    # locations[:, 1] = yBounds
    # boundaryVertices, boundaryNormals = calculate_boundary(locations)
    #
    # nTurbs = 50
    # turbineX,turbineY,d = make_start_grid(nTurbs,boundaryVertices,boundaryNormals)
    #
    # turbineZ = np.ones(nTurbs)*100.
    # rotorDiameter = np.ones(nTurbs)*130.
    # shearExp = 0.15
    #
    # num = 100
    # aep_slow = np.zeros(num)
    # aep_fast = np.zeros(num)
    #
    # d = np.linspace(-100.,100.,num)
    #
    # start_fast = time.time()
    # for j in range(num):
    #     x = np.zeros_like(turbineX)
    #     x[:] = turbineX[:]
    #     x[28] += d[j]
    #     aep_fast[j] = fast_calc_AEP(x, turbineY, turbineZ, rotorDiameter, windDirections, windSpeeds, windFrequencies,wakemodel='gaussian')
    #     aep_slow[j] = fast_calc_AEP(x, turbineY, turbineZ, rotorDiameter, windDirections, windSpeeds, windFrequencies,wakemodel='jensen')
    # time_fast = time.time()-start_fast

    # start_slow = time.time()
    # for i in range(num):
    #     print i
    #     x = np.zeros_like(turbineX)
    #     x[:] = turbineX[:]
    #     x[28] += d[i]
    #     aep_slow[i] = calcAEP(x,turbineY,turbineZ,rotorDiameter,windDirections,windSpeeds,windFrequencies,shearExp,wakemodel='gaussian')
    # time_slow = time.time()-start_slow
    #
    # print 'time to run fast: ', time_fast
    # print 'time to run slow: ', time_slow
    # print 'frac: ', time_slow/time_fast

    # plt.figure(1)
    # plt.plot(d,aep_slow)
    # plt.title('slow')
    #
    # plt.figure(2)
    # plt.plot(d,aep_fast)
    # plt.title('fast')

    # plt.figure()
    # plt.plot(d,aep_fast,'r',linewidth=2)
    # plt.plot(d,aep_slow,'--b')
    # plt.title('both')
    # plt.show()


    turbineX = np.array([0.,0.,0.,1000.,1000.,1000.,2000.])
    turbineY = np.array([0.,500.,1000.,0.,500.,1000.,1000.])
    turbineX = np.array([0.,0.,0.,0.])
    turbineY = np.array([0.,400.,800.,1200.])
    turbineZ = np.zeros(len(turbineX))+100.
    rotorDiameter = np.zeros(len(turbineX))+200.
    windDirections = np.array([0.])
    windSpeeds = np.array([10.])
    windFrequencies = np.array([1.])

    num = 1
    # y = np.linspace(-2000.,2000.,num)
    y = np.array([1000.])
    AEP = np.zeros(num)
    for i in range(num):
        turbineY[-1] = y[i]
        AEP[i] = calcAEP(turbineX,turbineY,turbineZ,rotorDiameter,windDirections,windSpeeds,windFrequencies,0.15,
                    wakemodel='floris',relaxationFactor=1.0)

    print AEP
    # import matplotlib.pyplot as plt
    # plt.plot(y,AEP,linewidth=2)
    # plt.show()
