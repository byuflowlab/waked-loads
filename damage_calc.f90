

subroutine amount_waked(dy,wake_radius,rotor_diameter,az,amnt_waked)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: dy, wake_radius, rotor_diameter, az

    ! out
    real(dp), intent(out) :: amnt_waked

    ! local
    real(dp) :: r, m, az_rad, b, d, dist, x1, x2, y1, y2, az_mod, d1, d2
    real(dp), dimension(3) :: p
    real(dp), dimension(2) :: dir_blade, dir_intersect1, dir_intersect2
    real(dp), parameter :: pi = 3.141592653589793_dp

    r = rotor_diameter/2.0
    az_rad = az/360.*2.*pi

    if (MOD(az,360.0) == 0. .OR. MOD(az,360.0) == 180.) then
        m = 1000.
    else
        m = cos(az_rad)/sin(az_rad)
    end if

    b = -m*dy
    p = (/m**2+1.,2.*m*b,b**2-wake_radius**2 /)

    d = p(2)**2 - 4.0*p(1)*p(3)
    if (d < 0.0) then
        amnt_waked = 0.
    else
        x1 = (-p(2)+SQRT(d))/(2.0*p(1))
        x2 = (-p(2)-SQRT(d))/(2.0*p(1))

        y1 = m*x1+b
        y2 = m*x2+b

        az_mod = 90.- MOD(az,360.)
        az_mod = az_mod/360.*2.*pi
        dir_blade = (/r*cos(az_mod),r*sin(az_mod) /)
        dir_intersect1 = (/x1-dy,y1 /)
        dir_intersect2 = (/x2-dy,y2 /)

        d1 = dir_blade(1)*dir_intersect1(1) + dir_blade(2)*dir_intersect1(2)
        d1 = dir_blade(1)*dir_intersect2(1) + dir_blade(2)*dir_intersect2(2)

        if (d1 < 0.0 .AND. d2 < 0.0) then
            amnt_waked = 0.
        else
            if (d1 < 0. .AND. d2 > 0.) then
                dist = SQRT((x2-dy)**2+(y2)**2)
            else if (d1 > 0.0 .AND. d2 < 0.0) then
                dist = SQRT((x1-dy)**2+(y1)**2)
            else
                if (d1 <= d2) then
                    dist = SQRT((x1-dy)**2+(y1)**2)
                else if (d2 < d1) then
                    dist = SQRT((x2-dy)**2+(y2)**2)
                end if
            end if

            if (ABS(dy) > wake_radius) then
                if (dist > r) then
                    amnt_waked = 0.0
                else
                    amnt_waked = 1.0-dist/r
                end if
            else
                if (dist > r) then
                    amnt_waked = 1.0
                else
                    amnt_waked = dist/r
                end if
            end if
        end if
    end if

end subroutine amount_waked















subroutine get_loads_history(nTurbines,NFast,NCC,turbineX,turbineY,turb_index,Omega_free,Omega_waked,free_speed,waked_speed,f_atm_free,f_atm_close,f_atm_far)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines, NFast, NCC, turb_index
    real(dp), dimension(nTurbines), intent(in) :: turbineX, turbineY
    real(dp), dimension(N), intent(in) :: atm_free, atm_close, atm_far
    real(dp), intent(in) :: Omega_free, Omega_waked, free_speed, waked_speed

    ! out
    real(dp), dimension(N), intent(out) :: moments

    ! local
    ! real(dp) :: r, m, az_rad, b, d, dist, x1, x2, y1, y2, az_mod, d1, d2
    ! real(dp), dimension(3) :: p
    ! real(dp), dimension(2) :: dir_blade, dir_intersect1, dir_intersect2
    ! real(dp), parameter :: pi = 3.141592653589793_dp


end subroutine get_loads_history

def get_loads_history(turbineX,turbineY,turb_index,Omega_free,Omega_waked,free_speed,waked_speed,f_atm_free,f_atm_close,f_atm_far,N=24001,TI=0.11,wind_speed=8.,rotor_diameter=126.4):


    Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,pitch,yaw_deg = setup_airfoil()
    angles = np.linspace(0.,360.,100)
    ccblade_moments = np.zeros_like(angles)

    # print 'getting CCBlade moments'
    """CCBlade moments"""
    s = Time.time()
    for i in range(len(ccblade_moments)):
        az = angles[i]
        x_locs,y_locs,z_locs = findXYZ(turbineX[turb_index],turbineY[turb_index],90.,r,yaw_deg,az)
        speeds, _ = get_speeds(turbineX, turbineY, x_locs, y_locs, z_locs, wind_speed,TI=TI)

        if i == 0:
            actual_speed = get_eff_turbine_speeds(turbineX, turbineY, wind_speed,TI=TI)[1]
            Omega = (Omega_waked + (Omega_free-Omega_waked)/(free_speed-waked_speed) * (actual_speed-waked_speed))

        ccblade_moments[i], _ = calc_moment(speeds,Rhub,r,chord,theta,af,Rhub,Rtip,B,rho,mu,precone,hubHt,nSector,Omega,pitch,azimuth=az)

    f_ccblade = interp1d(angles, ccblade_moments/1000., kind='linear')
    # print 'loop 1: ', Time.time()-s

    pos = np.linspace(0.,Omega*10.*360.,N)%360.

    """amount waked"""
    _, sigma = get_speeds(turbineX, turbineY, np.array([0.]), np.array([0.]), np.array([0.]), wind_speed, TI=TI)
    wake_radius = sigma*1.75

    waked_amount = np.zeros((nTurbines,N))
    dx_dist = np.zeros(nTurbines)




    for waking in range(nTurbines):
        dx = turbineX[turb_index]-turbineX[waking]
        dy = turbineY[turb_index]-turbineY[waking]
        dx_dist[waking] = dx
        amnt_waked = np.zeros(len(angles))
        for i in range(len(angles)):
            if dx > 0.:
                amnt_waked[i] = damage_calc.amount_waked(dy,wake_radius[turb_index][waking],rotor_diameter,angles[i])

        waked_func = interp1d(angles, amnt_waked, kind='linear')
        waked_amount[waking,:] = waked_func(pos)
        # print waked_func(pos)





    t = np.linspace(0.,600.,N)
    dt = t[1]-t[0]

    s = Time.time()


    moments = f_ccblade(pos)
    # moments = np.zeros(N)

    for i in range(N):
        # print '1'
        # s = Time.time()

        # amnt_waked = np.zeros(nTurbines)
        # dx_dist = np.zeros(nTurbines)
        # for waking in range(nTurbines):
        #     dx = turbineX[turb_index]-turbineX[waking]
        #     dy = turbineY[turb_index]-turbineY[waking]
        #     dx_dist[waking] = dx
        #     if dx < 0.:
        #         amnt_waked[waking] = 0.
        #     else:
        #         amnt_waked[waking] = amount_waked(dy,wake_radius[turb_index][waking],rotor_diameter,pos[i])


        # print i
        # print waked_array[:,i]
        # print 'shape: ', np.shape(waked_array)
        amnt_waked = waked_amount[:,i]



        waked_array = np.zeros(nTurbines)
        dx_array = np.zeros(nTurbines)

        # print 'loop 2: ', Time.time()-s
        # print '2'
        s = Time.time()

        num = 0
        indices = np.argsort(dx_dist)
        for waking in range(nTurbines):
            if dx_dist[indices[waking]] > 0.:
                if amnt_waked[indices[waking]] > np.sum(waked_array[0:num]):
                    waked_array[num] = amnt_waked[indices[waking]]-np.sum(waked_array[0:num])
                    dx_array[num] = dx_dist[indices[waking]]
                    num += 1

        # print Time.time()-s
        # print '3'
        s = Time.time()

        down = dx_array/rotor_diameter

        # moments[i] = f_ccblade(pos[i])

        unwaked = 1.-np.sum(waked_array)
        # print 'unwaked', unwaked
        for k in range(np.count_nonzero(waked_array)):
            if down[k] < 4.:
                  moments[i] += f_atm_close(t[i])*waked_array[k]
                  # moments[i] += atm_close[i]*waked_array[k]
            elif down[k] > 10.:
                  moments[i] += f_atm_far(t[i])*waked_array[k]
            else:
                  moments[i] += (f_atm_close(t[i])*(10.-down[k])/6.+f_atm_far(t[i])*(down[k]-4.)/6.)*waked_array[k]
        # print Time.time()-s
        moments[i] += f_atm_free(t[i])*unwaked

        # print 'pos: ', pos[i]
        # print 'position: ', position
        # position = (position+(Omega*(dt)/60.)*360.)%360.

    # print 'loop 2: ', Time.time()-s

    return moments
