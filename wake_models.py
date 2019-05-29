import numpy as np
import _floris
from scipy.interpolate import CubicSpline
import sys
sys.dont_write_bytecode = True


"""FLORIS"""
def Floris(turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf, yaw=False, Ct=False, kd=0.15, bd=-0.01, initialWakeDisplacement=-4.5,\
            useWakeAngle=False, initialWakeAngle=1.5, ke=0.065, adjustInitialWakeDiamToYaw=False, MU=np.array([0.5, 1.0, 5.5]),\
            useaUbU=True, aU=5.0, bU=1.66, me=np.array([-0.5, 0.22, 1.0]), cos_spread=2., Region2CT=0.888888888889, axialInduction=False, \
            keCorrCT=0.0, keCorrArray=0.0, axialIndProvided=True, shearCoefficientAlpha=0.10805, shearZh=90.):
# def Floris(turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf, yaw=False, Ct=False, kd=0.15, bd=-0.01, initialWakeDisplacement=-4.5,\
#             useWakeAngle=False, initialWakeAngle=1.5, ke=0.065, adjustInitialWakeDiamToYaw=False, MU=np.array([0.5, 1.0, 5.5]),\
#             useaUbU=True, aU=5.0, bU=1.66, me=np.array([-0.5, 0.22, 1.0]), cos_spread=1.e+12, Region2CT=0.888888888889, axialInduction=False, \
#             keCorrCT=0.0, keCorrArray=0.0, axialIndProvided=True, shearCoefficientAlpha=0.10805, shearZh=90.):
    """floris wake model"""
    nTurbines = len(turbineXw)

    if yaw == False:
        yaw = np.zeros(nTurbines)
    if axialInduction == False:
        axialInduction = np.ones(nTurbines)*1./3.
    if Ct == False:
        Ct = np.ones(nTurbines)*4.0*1./3.*(1.0-1./3.)

    # yaw wrt wind dir.
    yawDeg = yaw

    nSamples = 1
    wsPositionXYZw = np.zeros([3, nSamples])

    # call to fortran code to obtain output values
    wtVelocity, wsArray, wakeCentersYT, wakeCentersZT, wakeDiametersT, wakeOverlapTRel = \
                _floris.floris(turbineXw, turbineYw, turbineZ, yawDeg, rotorDiameter, Vinf,
                                               Ct, axialInduction, ke, kd, me, initialWakeDisplacement, bd,
                                               MU, aU, bU, initialWakeAngle, cos_spread, keCorrCT,
                                               Region2CT, keCorrArray, useWakeAngle,
                                               adjustInitialWakeDiamToYaw, axialIndProvided, useaUbU, wsPositionXYZw,
                                               shearCoefficientAlpha, shearZh)

    # return wtVelocity # , wakeCentersYT, wakeCentersZT, wakeDiametersT, wakeOverlapTRel
    return wtVelocity, wakeOverlapTRel


"""GAUS"""
def GaussianWake(turbineXw, turbineYw, turb_diam):
    """Return each turbine's total loss due to wake from upstream turbines"""
    # Equations and values explained in <iea37-wakemodel.pdf>
    num_turb = len(turbineXw)

    # Constant thrust coefficient
    CT = 4.0*1./3.*(1.0-1./3.)
    # Constant, relating to a turbulence intensity of 0.075
    k = 0.0324555
    # Array holding the wake deficit seen at each turbine
    loss = np.zeros(num_turb)

    for i in range(num_turb):            # Looking at each turb (Primary)
        loss_array = np.zeros(num_turb)  # Calculate the loss from all others
        for j in range(num_turb):        # Looking at all other turbs (Target)
            x = turbineXw[i] - turbineXw[j]   # Calculate the x-dist
            y = turbineYw[i] - turbineYw[j]   # And the y-offset
            if x > 0.:                   # If Primary is downwind of the Target
                sigma = k*x + turb_diam/np.sqrt(8.)  # Calculate the wake loss
                # Simplified Bastankhah Gaussian wake model
                exponent = -0.5 * (y/sigma)**2
                radical = 1. - CT/(8.*sigma**2 / turb_diam**2)
                loss_array[j] = (1.-np.sqrt(radical)) * np.exp(exponent)
            # Note that if the Target is upstream, loss is defaulted to zero
        # Total wake losses from all upstream turbs, using sqrt of sum of sqrs
        loss[i] = np.sqrt(np.sum(loss_array**2))
    # print '----: ', loss[0]
    return loss


def Gaussian(turbineXw, turbineYw, turb_diam, wind_speed):
    loss = GaussianWake(turbineXw, turbineYw, turb_diam)
    wtVelocity = wind_speed*(1.-loss)
    return wtVelocity


"""JENSEN"""
def get_cosine_factor_original(X, Y, R0, bound_angle=20.0, relaxationFactor=1.0):

    n = np.size(X)
    bound_angle = bound_angle*np.pi/180.0           # convert bound_angle from degrees to radians
    # theta = np.zeros((n, n), dtype=np.float)      # angle of wake from fulcrum
    f_theta = np.zeros((n, n), dtype=np.float)      # smoothing values for smoothing
    q = np.pi/bound_angle                           # factor inside the cos term of the smooth Jensen (see Jensen1983 eq.(3))

    # Idea for relaxation factor requires new angle, gamma. Units in radians.
    gamma = (np.pi/2.0) - bound_angle

    for i in range(0, n):
        for j in range(0, n):

            if X[i] < X[j]:
                z = (relaxationFactor * R0 * np.sin(gamma))/np.sin(bound_angle)
                theta = np.arctan((Y[j] - Y[i]) / (X[j] - X[i] + z))
                if -bound_angle < theta < bound_angle:
                    f_theta[i][j] = (1. + np.cos(q*theta))/2.

    return f_theta


def JensenWake(turbineXw, turbineYw, turb_diam, relaxationFactor = 1.0):
    """Return each turbine's total loss due to wake from upstream turbines"""
    # Equations and values explained in <iea37-wakemodel.pdf>
    num_turb = len(turbineXw)

    # Constant axial induction
    a = 1./3.

    alpha = 0.1
    r0 = turb_diam/2.
    # theta =
    # Array holding the wake deficit seen at each turbine
    loss = np.zeros(num_turb)

    f_theta = get_cosine_factor_original(turbineXw, turbineYw, R0=r0, relaxationFactor=relaxationFactor)

    for i in range(num_turb):            # Looking at each turb (Primary)
        loss_array = np.zeros(num_turb)  # Calculate the loss from all others
        for j in range(num_turb):        # Looking at all other turbs (Target)
            x = turbineXw[i] - turbineXw[j]   # Calculate the x-dist
            y = turbineYw[i] - turbineYw[j]   # And the y-offset
            if x > 0.:                   # If Primary is downwind of the Target
                r = alpha*x + r0
                # if abs(y) <= r:
                # if abs(y) <= r:
                    # loss_array[j] = 2.*a*(r0/(r0 + alpha*x))**2
                loss_array[j] = 2.*a*(r0*f_theta[j][i]/(r0 + alpha*x))**2
        loss[i] = np.sqrt(np.sum(loss_array**2))

    return loss


def Jensen(turbineXw, turbineYw, turb_diam, wind_speed, relaxationFactor=1.0):
    loss = JensenWake(turbineXw, turbineYw, turb_diam, relaxationFactor=relaxationFactor)
    wtVelocity = wind_speed*(1.-loss)
    return wtVelocity
