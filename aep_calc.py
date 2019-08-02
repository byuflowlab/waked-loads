import numpy as np
import gaus
import sys
import fast_calc_aep
from wakeexchange.utilities import sunflower_points
sys.dont_write_bytecode = True


def WindDirectionPower(wtVelocity,rated_ws=False,rated_power=False,cut_in_speed=False,cut_out_speed=False):
    """calculate power from a given wind direction"""
    nTurbines = len(wtVelocity)

    if rated_ws == False:
        rated_ws = 11.4
    if rated_power == False:
        rated_power = 5.
    if cut_in_speed == False:
        cut_in_speed = 3.
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


def calcAEP(turbineX,turbineY,windDirections,windSpeeds,windFrequencies,TI=0.11,relaxationFactor=1.0):

    nTurbines = len(turbineX)
    turbineZ = np.ones(nTurbines)*90.
    yaw = np.zeros(nTurbines)
    rotorDiameter = np.ones(nTurbines)*126.4
    ky = 0.022
    kz = 0.022
    alpha = 2.32
    beta = 0.154
    z_ref = 50.
    z_0 = 0.
    # RotorPointsY = np.array([0.])
    # RotorPointsZ = np.array([0.])
    nRotorPoints = 20
    RotorPointsY, RotorPointsZ = sunflower_points(nRotorPoints)

    sorted_x_idx = np.argsort(turbineX)

    use_ct_curve = False
    interp_type = 1.
    Ct = np.ones(nTurbines)*8./9.

    ct_curve_ct = Ct

    wake_model_version=2016
    sm_smoothing=700.
    calc_k_star=True
    ti_calculation_method=2
    wake_combination_method=1
    print_ti = False

    shear_exp=0.
    TI=0.11

    nDirections = len(windDirections)

    dir_powers = np.zeros(nDirections)

    for i in range(nDirections):
        ct_curve_wind_speed = np.ones_like(Ct)*windSpeeds[i]

        turbineXw, turbineYw = fast_calc_aep.windframe(windDirections[i], turbineX, turbineY)
        wtVelocity = gaus.porteagel_analyze(turbineX,sorted_x_idx,turbineY,
                            turbineZ,rotorDiameter,Ct,windSpeeds[i],yaw,ky,kz,alpha,beta,
                            TI,RotorPointsY,RotorPointsZ,z_ref,z_0,shear_exp,wake_combination_method,
                            ti_calculation_method,calc_k_star,relaxationFactor,print_ti,wake_model_version,
                            interp_type,use_ct_curve,ct_curve_wind_speed,ct_curve_ct,sm_smoothing)

        _,dir_powers[i] = WindDirectionPower(wtVelocity)

    AEP = np.sum(dir_powers*windFrequencies*3600.*24.*365.)

    return AEP


if __name__=="__main__":


    windDirections = np.array([270.])
    windSpeeds = np.array([8.])
    windFrequencies = np.array([1.])
    loc = np.linspace(-500.,500.,1000)
    AEP300 = np.zeros_like(loc)
    AEP500 = np.zeros_like(loc)
    AEP800 = np.zeros_like(loc)

    import time
    import matplotlib.pyplot as plt
    s = time.time()

    turbineX = np.array([0.,300])
    AEP1 = calcAEP(turbineX,np.array([0.,loc[0]]),windDirections,windSpeeds,windFrequencies)

    for i in range(1000):
        AEP300[i] = calcAEP(turbineX,np.array([0.,loc[i]]),windDirections,windSpeeds,windFrequencies)-AEP1/2.

    turbineX = np.array([0.,500])
    for i in range(1000):
        AEP500[i] = calcAEP(turbineX,np.array([0.,loc[i]]),windDirections,windSpeeds,windFrequencies)-AEP1/2.

    turbineX = np.array([0.,800])
    for i in range(1000):
        AEP800[i] = calcAEP(turbineX,np.array([0.,loc[i]]),windDirections,windSpeeds,windFrequencies)-AEP1/2.

    print (time.time()-s)/3000.
    plt.ylim(0.,6.E7)
    plt.plot(loc,AEP300)
    plt.plot(loc,AEP500)
    plt.plot(loc,AEP800)
    plt.show()
