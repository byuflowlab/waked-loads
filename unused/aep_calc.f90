
!global functions
subroutine WindFrame(nTurbines, wind_direction, turbineX, turbineY, turbineXw, turbineYw)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines
    real(dp), intent(in) :: wind_direction
    real(dp), dimension(nTurbines), intent(in) :: turbineX, turbineY

    ! out
    real(dp), dimension(nTurbines), intent(out) :: turbineXw, turbineYw

    ! local
    real(dp) :: windDirectionDeg, windDirectionRad
    real(dp), parameter :: pi = 3.141592653589793_dp, tol = 0.000001_dp

    windDirectionDeg = 270. - wind_direction
    if (windDirectionDeg < 0.) then
        windDirectionDeg = windDirectionDeg + 360.
    end if
    windDirectionRad = pi*windDirectionDeg/180.0

    turbineXw = turbineX*cos(-windDirectionRad)-turbineY*sin(-windDirectionRad)
    turbineYw = turbineX*sin(-windDirectionRad)+turbineY*cos(-windDirectionRad)

end subroutine WindFrame


subroutine Hermite_Spline(x, x0, x1, y0, dy0, y1, dy1, y)
    !    This function produces the y and dy values for a hermite cubic spline
    !    interpolating between two end points with known slopes
    !
    !    :param x: x position of output y
    !    :param x0: x position of upwind endpoint of spline
    !    :param x1: x position of downwind endpoint of spline
    !    :param y0: y position of upwind endpoint of spline
    !    :param dy0: slope at upwind endpoint of spline
    !    :param y1: y position of downwind endpoint of spline
    !    :param dy1: slope at downwind endpoint of spline
    !
    !    :return: y: y value of spline at location x

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: x, x0, x1, y0, dy0, y1, dy1

    ! out
    real(dp), intent(out) :: y !, dy_dx

    ! local
    real(dp) :: c3, c2, c1, c0

    ! initialize coefficients for parametric cubic spline
    c3 = (2.0_dp*(y1))/(x0**3 - 3.0_dp*x0**2*x1 + 3.0_dp*x0*x1**2 - x1**3) - &
         (2.0_dp*(y0))/(x0**3 - 3.0_dp*x0**2*x1 + 3.0_dp*x0*x1**2 - x1**3) + &
         (dy0)/(x0**2 - 2.0_dp*x0*x1 + x1**2) + &
         (dy1)/(x0**2 - 2.0_dp*x0*x1 + x1**2)

    c2 = (3.0_dp*(y0)*(x0 + x1))/(x0**3 - 3.0_dp*x0**2*x1 + 3.0_dp*x0*x1**2 - x1**3) - &
         ((dy1)*(2.0_dp*x0 + x1))/(x0**2 - 2.0_dp*x0*x1 + x1**2) - ((dy0)*(x0 + &
         2.0_dp*x1))/(x0**2 - 2.0_dp*x0*x1 + x1**2) - (3.0_dp*(y1)*(x0 + x1))/(x0**3 - &
         3.0_dp*x0**2*x1 + 3.0_dp*x0*x1**2 - x1**3)

    c1 = ((dy0)*(x1**2 + 2.0_dp*x0*x1))/(x0**2 - 2.0_dp*x0*x1 + x1**2) + ((dy1)*(x0**2 + &
         2.0_dp*x1*x0))/(x0**2 - 2.0_dp*x0*x1 + x1**2) - (6.0_dp*x0*x1*(y0))/(x0**3 - &
         3.0_dp*x0**2*x1 + 3.0_dp*x0*x1**2 - x1**3) + (6.0_dp*x0*x1*(y1))/(x0**3 - &
         3.0_dp*x0**2*x1 + 3.0_dp*x0*x1**2 - x1**3)

    c0 = ((y0)*(- x1**3 + 3.0_dp*x0*x1**2))/(x0**3 - 3.0_dp*x0**2*x1 + 3.0_dp*x0*x1**2 - &
         x1**3) - ((y1)*(- x0**3 + 3.0_dp*x1*x0**2))/(x0**3 - 3.0_dp*x0**2*x1 + &
         3.0_dp*x0*x1**2 - x1**3) - (x0*x1**2*(dy0))/(x0**2 - 2.0_dp*x0*x1 + x1**2) - &
         (x0**2*x1*(dy1))/(x0**2 - 2.0_dp*x0*x1 + x1**2)

    ! Solve for y and dy values at the given point
    y = c3*x**3 + c2*x**2 + c1*x + c0
    !dy_dx = c3*3*x**2 + c2*2*x + c1

end subroutine Hermite_Spline

!
! !power calculations"
subroutine PowWind(nTurbines, Uref, turbineZ, shearExp, zref, z0, &
                    &turbineSpeeds)

    implicit none
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)
    ! in
    integer, intent(in) :: nTurbines
    real(dp), intent(in) :: Uref, shearExp, zref, z0
    real(dp), dimension(nTurbines), intent(in) :: turbineZ
    ! out
    real(dp), dimension(nTurbines), intent(out) :: turbineSpeeds
    ! local
    integer :: n

    do n = 1, nTurbines
        turbineSpeeds(n)= Uref*((turbineZ(n)-z0)/(zref-z0))**shearExp
    end do

end subroutine PowWind


subroutine DirPower(nTurbines, wtVelocity, rated_ws, rated_power, cut_in_speed, cut_out_speed,&
                        &dir_power)
    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines
    real(dp), intent(in) :: rated_ws, rated_power, cut_in_speed, cut_out_speed
    real(dp), dimension(nTurbines), intent(in) :: wtVelocity

    ! out
    real(dp), intent(out) :: dir_power

    ! local
    real(dp), dimension(nTurbines) :: wtPower
    real(dp) :: buffer, x0, x1, y0, y1, dy0, dy1
    integer :: n

    buffer = 0.1

    do n = 1, nTurbines
        ! If we're below cut-in
        if (wtVelocity(n) < (cut_in_speed-buffer)) then
            wtPower(n) = 0.
        ! If we're at the spline of cut-in
        else if (wtVelocity(n) > (cut_in_speed-buffer) .and. (wtVelocity(n) < (cut_in_speed+buffer))) then
            x0 = cut_in_speed-buffer
            x1 = cut_in_speed+buffer
            y0 = 0.
            y1 = rated_power*((cut_in_speed+buffer)/rated_ws)**3
            dy0 = 0.
            dy1 = 3.*rated_power*(cut_in_speed+buffer)**2/(rated_ws**3)
            call Hermite_Spline(wtVelocity(n), x0, x1, y0, dy0, y1, dy1, wtPower(n))
        ! If we're between cut-in and rated
        else if ((wtVelocity(n) > (cut_in_speed+buffer)) .and. (wtVelocity(n) < (rated_ws-buffer))) then
            wtPower(n) = rated_power*(wtVelocity(n)/rated_ws)**3
        ! If we're at the spline of rated
        else if ((wtVelocity(n) > (rated_ws-buffer)) .and. (wtVelocity(n) < (rated_ws+buffer))) then
            x0 = rated_ws-buffer
            x1 = rated_ws+buffer
            y0 = rated_power*((rated_ws-buffer)/rated_ws)**3
            y1 = rated_power
            dy0 = 3.*rated_power*(rated_ws-buffer)**2/(rated_ws**3)
            dy1 = 0.
            call Hermite_Spline(wtVelocity(n), x0, x1, y0, dy0, y1, dy1, wtPower(n))
        ! If we're between rated and cut-out
        else if ((wtVelocity(n) > (rated_ws+buffer)) .and. (wtVelocity(n) < (cut_out_speed-buffer))) then
            wtPower(n) = rated_power
        ! If we're at the spline of cut-out
        else if ((wtVelocity(n) > (cut_out_speed-buffer)) .and. (wtVelocity(n) < (cut_out_speed+buffer))) then
            x0 = cut_out_speed-buffer
            x1 = cut_out_speed+buffer
            y0 = rated_power
            y1 = 0.
            dy0 = 0.
            dy1 = 0.
            call Hermite_Spline(wtVelocity(n), x0, x1, y0, dy0, y1, dy1, wtPower(n))
        ! If we're above cut-out
        else if (wtVelocity(n) > (cut_out_speed+buffer)) then
            wtPower(n) = 0.
        end if

    end do

    dir_power = sum(wtPower)

end subroutine DirPower


subroutine calcAEP(nTurbines, nDirections, turbineX, turbineY, turbineZ, rotorDiameter, windDirections,&
            &windSpeeds, windFrequencies, shearExp, wakemodel, relaxationFactor, rated_ws, rated_power,&
            &cut_in_speed, cut_out_speed, zref, z0, AEP)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines, nDirections, wakemodel
    real(dp), intent(in) :: shearExp, relaxationFactor, rated_ws, rated_power, cut_in_speed, cut_out_speed, zref, z0
    real(dp), dimension(nTurbines), intent(in) :: turbineX, turbineY, turbineZ, rotorDiameter
    real(dp), dimension(nDirections), intent(in) :: windDirections, windSpeeds, windFrequencies

    ! out
    real(dp), intent(out) :: AEP

    ! local
    real(dp), dimension(nDirections) :: dir_powers
    real(dp), dimension(nTurbines) :: turbineXw, turbineYw, Vinf_floris, wtVelocity, loss
    real(dp) :: hrs_per_year, pwrDir, Vinf
    integer :: n, i

    ! print *, "got into the function"
    ! print *, wakemodel
    do n = 1, nDirections
        call WindFrame(nTurbines, windDirections(n), turbineX, turbineY, turbineXw, turbineYw)
        call PowWind(nTurbines, windSpeeds(n), turbineZ, shearExp, zref, z0, Vinf_floris)
        ! do i = 1, nTurbines
        !     Vinf_floris(i) = Vinf
        ! end do
        Vinf = Vinf_floris(1)

        if (wakemodel == 1) then
            call JensenWake(nTurbines, turbineXw, turbineYw, rotorDiameter(1), relaxationFactor, loss)
        else if (wakemodel == 2) then
            call GaussianWake(nTurbines, turbineXw, turbineYw, rotorDiameter(1), relaxationFactor, loss)
        else if (wakemodel == 3) then
            ! print *, Vinf_floris
            print *, 'floris'
            call FlorisWake(nTurbines, turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf_floris, wtVelocity, loss)
            ! print *, wtVelocity(1)

        end if

        ! print *, "loss: ", loss
        ! print *, Vinf
        wtVelocity = Vinf*(1.0_dp-loss)
        ! print *, wtVelocity(1)

        call DirPower(nTurbines, wtVelocity, rated_ws, rated_power, cut_in_speed, cut_out_speed, pwrDir)
        dir_powers(n) = pwrDir

    end do

    hrs_per_year = 365.*24.
    AEP = hrs_per_year * (sum(windFrequencies * dir_powers))

end subroutine calcAEP



!wake models"
!jensen"
subroutine get_cosine_factor_original(nTurbines, X, Y, R0, bound_angle, relaxationFactor,f_theta)

    implicit none
    integer, parameter :: dp = kind(0.d0)
    integer, intent(in) :: nTurbines

    real(dp), intent(in) :: bound_angle, R0, relaxationFactor
    real(dp), dimension(nTurbines), intent(in) :: X, Y

    real(dp), parameter :: pi = 3.141592653589793_dp

    real(dp), dimension(nTurbines,nTurbines), intent(out) :: f_theta

    real(dp) :: q, gamma, theta, z
    integer :: i, j

    q = pi/(bound_angle*pi/180.0)

    gamma = pi/2.0 - (bound_angle*pi/180.0)

    do i = 1, nTurbines
        do j = 1, nTurbines
            if (X(i) < X(j)) then
                z = (relaxationFactor * R0 * sin(gamma))/sin((bound_angle*pi/180.0))
                theta = atan((Y(j) - Y(i)) / (X(j) - X(i) + z))
                if (-(bound_angle*pi/180.0) < theta .and. theta < (bound_angle*pi/180.0)) then
                    f_theta(i,j) = (1. + cos(q*theta))/2.
                else
                    f_theta(i,j) = 0.
                end if
            else
                f_theta(i,j) = 0.
            end if
        end do
    end do

end subroutine get_cosine_factor_original


subroutine JensenWake(nTurbines, turbineXw, turbineYw, turb_diam, relaxationFactor, loss)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines
    real(dp), intent(in) :: turb_diam, relaxationFactor
    real(dp), dimension(nTurbines), intent(in) :: turbineXw, turbineYw

    ! out
    real(dp), dimension(nTurbines), intent(out) :: loss

    ! local
    real(dp) :: a, alpha, r0, bound_angle, x, y, r
    real(dp), dimension(nTurbines) :: loss_array
    real(dp), dimension(nTurbines,nTurbines) :: f_theta
    real(dp), parameter :: pi = 3.141592653589793_dp
    integer :: i, j

    a = 1./3.
    alpha = 0.1
    r0 = turb_diam/2.
    bound_angle = 20.

    call get_cosine_factor_original(nTurbines, turbineXw, turbineYw, r0, bound_angle, relaxationFactor, f_theta)

    do i = 1, nTurbines
        do j = 1, nTurbines
            x = turbineXw(i) - turbineXw(j)
            y = turbineYw(i) - turbineYw(j)
            if (x > 0.) then
                r = alpha*x + r0
                loss_array(j) = 2.*a*(r0*f_theta(j,i)/(r0 + alpha*x))**2
            else
                loss_array(j) = 0.
            end if
        end do
        loss(i) = sqrt(sum(loss_array**2))
    end do

end subroutine JensenWake

!gaus"
subroutine GaussianWake(nTurbines, turbineXw, turbineYw, turb_diam, relaxationFactor, loss)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines
    real(dp), intent(in) :: turb_diam, relaxationFactor
    real(dp), dimension(nTurbines), intent(in) :: turbineXw, turbineYw

    ! out
    real(dp), dimension(nTurbines), intent(out) :: loss

    ! local
    real(dp) :: CT, k, x, y, sigma, exponent, radical
    real(dp), dimension(nTurbines) :: loss_array
    real(dp), parameter :: pi = 3.141592653589793_dp, e = 2.718281828459045_dp, tol = 0.000001_dp
    integer :: i, j

    CT = 4.0*1./3.*(1.0-1./3.)
    k = 0.0324555

    do i = 1, nTurbines
        do j = 1, nTurbines
            x = turbineXw(i) - turbineXw(j)
            y = turbineYw(i) - turbineYw(j)
            if (x > 0.0) then
                sigma = k*x + turb_diam/sqrt(8.0)
                exponent = -0.5 * (y/(relaxationFactor*sigma))**2
                radical = 1. - CT/(8.*sigma**2 / turb_diam**2)
                loss_array(j) = (1.-sqrt(radical)) * e**exponent
            else
                loss_array(j) = 0.0
            end if
        end do
        loss(i) = sqrt(sum(loss_array**2))
    end do

    ! print *, loss(1)

end subroutine GaussianWake


! !floris"
subroutine calcOverlapAreas(nTurbines, turbineX, turbineY, turbineZ, rotorDiameter, wakeDiameters, &
                            wakeCentersYT, wakeCentersZT, wakeOverlapTRel_mat)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines
    real(dp), dimension(nTurbines), intent(in) :: turbineX, turbineY, turbineZ, rotorDiameter
    real(dp), dimension(nTurbines, nTurbines, 3), intent(in) :: wakeDiameters
    real(dp), dimension(nTurbines, nTurbines), intent(in) :: wakeCentersYT, wakeCentersZT

    ! out
    real(dp), dimension(nTurbines, nTurbines, 3), intent(out) :: wakeOverlapTRel_mat

    ! local
    integer :: turb, turbI, zone
    real(dp), parameter :: pi = 3.141592653589793_dp, tol = 0.000001_dp
    real(dp) :: OVdYd, OVr, OVRR, OVL, OVz
    real(dp), dimension(nTurbines, nTurbines, 3) :: wakeOverlap

    wakeOverlapTRel_mat = 0.0_dp
    wakeOverlap = 0.0_dp

    do turb = 1, nTurbines
        do turbI = 1, nTurbines
            if (turbineX(turbI) > turbineX(turb)) then
                OVdYd = sqrt((wakeCentersYT(turbI, turb)-turbineY(turbI))**2+(wakeCentersZT(turbI, turb)-turbineZ(turbI))**2)     ! distance between wake center and rotor center
                OVr = rotorDiameter(turbI)/2                        ! rotor diameter
                do zone = 1, 3
                    OVRR = wakeDiameters(turbI, turb, zone)/2.0_dp        ! wake diameter
                    OVdYd = abs(OVdYd)
                    if (OVdYd >= 0.0_dp + tol) then
                        ! calculate the distance from the wake center to the vertical line between
                        ! the two circle intersection points
                        OVL = (-OVr*OVr+OVRR*OVRR+OVdYd*OVdYd)/(2.0_dp*OVdYd)
                    else
                        OVL = 0.0_dp
                    end if

                    OVz = OVRR*OVRR-OVL*OVL

                    ! Finish calculating the distance from the intersection line to the outer edge of the wake zone
                    if (OVz > 0.0_dp + tol) then
                        OVz = sqrt(OVz)
                    else
                        OVz = 0.0_dp
                    end if

                    if (OVdYd < (OVr+OVRR)) then ! if the rotor overlaps the wake zone

                        if (OVL < OVRR .and. (OVdYd-OVL) < OVr) then
                            wakeOverlap(turbI, turb, zone) = OVRR*OVRR*dacos(OVL/OVRR) + OVr*OVr*dacos((OVdYd-OVL)/OVr) - OVdYd*OVz
                        else if (OVRR > OVr) then
                            wakeOverlap(turbI, turb, zone) = pi*OVr*OVr
                        else
                            wakeOverlap(turbI, turb, zone) = pi*OVRR*OVRR
                        end if
                    else
                        wakeOverlap(turbI, turb, zone) = 0.0_dp
                    end if

                end do

            end if

        end do

    end do


    do turb = 1, nTurbines

        do turbI = 1, nTurbines

            wakeOverlap(turbI, turb, 3) = wakeOverlap(turbI, turb, 3)-wakeOverlap(turbI, turb, 2)
            wakeOverlap(turbI, turb, 2) = wakeOverlap(turbI, turb, 2)-wakeOverlap(turbI, turb, 1)

        end do

    end do

    wakeOverlapTRel_mat = wakeOverlap

    do turbI = 1, nTurbines
            wakeOverlapTRel_mat(turbI, :, :) = wakeOverlapTRel_mat(turbI, :, &
                                                         :)/((pi*rotorDiameter(turbI) &
                                                       *rotorDiameter(turbI))/4.0_dp)
    end do

end subroutine calcOverlapAreas


subroutine CTtoAxialInd(CT, nTurbines, axial_induction)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines
    real(dp), dimension(nTurbines), intent(in) :: CT

    ! local
    integer :: i

    ! out
    real(dp), dimension(nTurbines), intent(out) :: axial_induction

    axial_induction = 0.0_dp

    ! execute
    do i = 1, nTurbines
        if (CT(i) > 0.96) then  ! Glauert condition
            axial_induction(i) = 0.143_dp + sqrt(0.0203_dp-0.6427_dp*(0.889_dp - CT(i)))
        else
            axial_induction(i) = 0.5_dp*(1.0_dp-sqrt(1.0_dp-CT(i)))
        end if
    end do

end subroutine CTtoAxialInd


subroutine FlorisWake(nTurbines, turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf, wtVelocity, loss)

    ! independent variables: yawDeg Ct turbineXw turbineYw turbineZ  rotorDiameter a_in
    ! dependent variables: wtVelocity

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines
    real(dp), dimension(nTurbines), intent(in) :: turbineXw, turbineYw, turbineZ, rotorDiameter, Vinf

    ! local (General)
    real(dp), dimension(nTurbines) :: ke, yaw
    real(dp) :: deltax
    Integer :: turb, turbI, zone, i
    real(dp), parameter :: pi = 3.141592653589793_dp


    ! local (Wake centers and diameters)
    real(dp) :: spline_bound ! in rotor diameters
    real(dp) :: wakeAngleInit, zeroloc
    real(dp) :: factor, displacement, x, x1, x2, y1, y2, dy1, dy2
    real(dp) :: wakeDiameter0
    real(dp), dimension(nTurbines, nTurbines, 3) :: wakeDiametersT_mat
    real(dp), dimension(nTurbines, nTurbines) :: wakeCentersYT_mat, wakeCentersZT_mat

    ! local (Wake overlap)
    real(dp) :: rmax
    real(dp), dimension(nTurbines, nTurbines, 3) :: wakeOverlapTRel_mat

    ! local (Velocity)
    real(dp), dimension(nTurbines) :: a, keArray
    real(dp), dimension(3) :: mmU
    real(dp) :: s, cosFac, wakeEffCoeff, wakeEffCoeffPerZone

    ! local (tuning parameters)
    real(dp) :: kd, bd, initialWakeDisplacement, initialWakeAngle, ke_in, aU, bU, cos_spread, Region2CT, keCorrCT, keCorrArray
    logical :: useWakeAngle, adjustInitialWakeDiamToYaw, useaUbU, axialIndProvided
    real(dp), dimension(3) :: MU, me
    real(dp), dimension(nTurbines) :: yawDeg, Ct, a_in

    ! model out
    real(dp), dimension(nTurbines), intent(out) :: wtVelocity, loss

    intrinsic cos, atan, max

    kd = 0.15
    bd = -0.01
    initialWakeDisplacement = -4.5
    initialWakeAngle = 1.5
    ke_in = 0.065
    aU = 5.0
    bU = 1.66
    cos_spread = 2.0
    Region2CT = 0.888888888889
    keCorrCT = 0.
    keCorrArray = 0.

    useWakeAngle = .false.
    adjustInitialWakeDiamToYaw = .false.
    useaUbU = .true.
    axialIndProvided = .true.

    MU(1) = 0.5
    MU(2) = 1.0
    MU(3) = 5.5

    me(1) = -0.5
    me(2) = 0.22
    me(3) = 1.0

    do i = 1, nTurbines
        yawDeg(i) = 0.
        a_in(i) = 1./3.
        Ct(i) = 4.0*1./3.*(1.0-a_in(i))
    end do


    yaw = yawDeg*pi/180.0_dp

    !!!!!!!!!!!!!!!!!!!!!!!!!!!! Wake Centers and Diameters !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    spline_bound = 1.0_dp

    ! calculate locations of wake centers in wind ref. frame
    wakeCentersYT_mat = 0.0_dp
    wakeCentersZT_mat = 0.0_dp

    do turb = 1, nTurbines
        wakeAngleInit = 0.5_dp*sin(yaw(turb))*Ct(turb)

        if (useWakeAngle) then
            wakeAngleInit = wakeAngleInit + initialWakeAngle*pi/180.0_dp
        end if

        ! wake center calculations at each turbine
        do turbI = 1, nTurbines

            if (turbineXw(turb) < turbineXw(turbI)) then
                deltax = turbineXw(turbI) - turbineXw(turb)
                factor = (2.0_dp*kd*deltax/rotorDiameter(turb)) + 1.0_dp

                !THESE ARE THE Z CALCULATIONS
                wakeCentersZT_mat(turbI, turb) = turbineZ(turb)

                !THESE ARE THE Y CALCULATIONS
                wakeCentersYT_mat(turbI, turb) = turbineYw(turb)

                displacement = wakeAngleInit*(wakeAngleInit* &
                                                 & wakeAngleInit + 15.0_dp*factor*factor* &
                                                 factor*factor)/((30.0_dp*kd/ &
                                                 rotorDiameter(turb))*(factor*factor* &
                                                 & factor*factor*factor))

                displacement = displacement - &
                                                 & wakeAngleInit*(wakeAngleInit* &
                                                 & wakeAngleInit + 15.0_dp)/(30.0_dp*kd/ &
                                                 rotorDiameter(turb))

                wakeCentersYT_mat(turbI, turb) = wakeCentersYT_mat(turbI, turb)+ &
                                                 & initialWakeDisplacement + displacement

                if (useWakeAngle .eqv. .false.) then
                    wakeCentersYT_mat(turbI, turb) = wakeCentersYT_mat(turbI, turb) + bd*(deltax)
                end if

            end if

        end do

    end do

    !adjust k_e to C_T, adjusted to yaw
    ke = ke_in + keCorrCT*(Ct-Region2CT)

    ! calculate wake diameters
    wakeDiametersT_mat = 0.0_dp

    do turb = 1, nTurbines

        if (adjustInitialWakeDiamToYaw) then
            wakeDiameter0 = rotorDiameter(turb)*cos(yaw(turb))
        else
            wakeDiameter0 = rotorDiameter(turb)
        end if

        ! calculate the wake diameter of each wake at each turbine
        do turbI = 1, nTurbines

            ! turbine separation
            deltax = turbineXw(turbI) - turbineXw(turb)

            ! x position of interest
            x = turbineXw(turbI)

            zone = 1

            ! define centerpoint of spline
            zeroloc = turbineXw(turb) - wakeDiameter0/(2.0_dp*ke(turb)*me(zone))

            if (zeroloc + spline_bound*rotorDiameter(turb) < turbineXw(turbI)) then ! check this
                wakeDiametersT_mat(turbI, turb, zone) = 0.0_dp

            else if (zeroloc - spline_bound*rotorDiameter(turb) < turbineXw(turbI)) then !check this

                !!!!!!!!!!!!!!!!!!!!!! calculate spline values !!!!!!!!!!!!!!!!!!!!!!!!!!

                ! position of upwind point
                x1 = zeroloc - spline_bound*rotorDiameter(turb)
                ! diameter of upwind point
                y1 = wakeDiameter0+2.0_dp*ke(turb)*me(zone)*(x1 - turbineXw(turb))
                ! slope at upwind point
                dy1 = 2.0_dp*ke(turb)*me(zone)
                ! position of downwind point
                x2 = zeroloc+spline_bound*rotorDiameter(turb)
                ! diameter at downwind point
                y2 = 0.0_dp
                ! slope at downwind point
                dy2 = 0.0_dp

                ! solve for the wake zone diameter and its derivative w.r.t. the downwind
                ! location at the point of interest
                call Hermite_Spline(x, x1, x2, y1, dy1, y2, dy2, wakeDiametersT_mat(turbI, turb, zone))

            else if (turbineXw(turb) < turbineXw(turbI)) then
                wakeDiametersT_mat(turbI, turb, zone) = wakeDiameter0+2.0_dp*ke(turb)*me(zone)*deltax
            end if


            if (turbineXw(turb) < turbineXw(turbI)) then
                zone = 2
                wakeDiametersT_mat(turbI, turb, zone) = wakeDiameter0 + 2.0_dp*ke(turb)*me(zone)*deltax
                zone = 3
                wakeDiametersT_mat(turbI, turb, zone) = wakeDiameter0 + 2.0_dp*ke(turb)*me(zone)*deltax
            end if

        end do

    end do

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Wake Overlap !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ! calculate relative overlap
    call calcOverlapAreas(nTurbines, turbineXw, turbineYw, turbineZ, rotorDiameter, &
                          & wakeDiametersT_mat, wakeCentersYT_mat, wakeCentersZT_mat, wakeOverlapTRel_mat)

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Velocity !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ! initialize axial induction values
    if (axialIndProvided) then
        a = a_in
    else
        call CTtoAxialInd(Ct, nTurbines, a)
    end if

    ! adjust ke to Ct as adjusted to yaw
    ke = ke_in + keCorrCT*(Ct-Region2CT)

    do turb = 1, nTurbines
        s = sum(wakeOverlapTRel_mat(turb, :, 1) + wakeOverlapTRel_mat(turb, :, 2))
        keArray(turb) = ke(turb)*(1+s*keCorrArray)
    end do

    ! find effective wind speeds at downstream turbines
    wtVelocity = Vinf
    do turbI = 1, nTurbines
        wakeEffCoeff = 0.0_dp

        ! find overlap-area weighted effect of each wake zone
        do turb = 1, nTurbines
            wakeEffCoeffPerZone = 0.0_dp
            deltax = turbineXw(turbI) - turbineXw(turb)

            if (useaUbU) then
                mmU = MU/cos(aU*pi/180.0_dp + bU*yaw(turb))
            end if

            if (deltax > 0 .and. turbI /= turb) then
                do zone = 1, 3

                    rmax = cos_spread*0.5_dp*(wakeDiametersT_mat(turbI, turb, 3) + rotorDiameter(turbI))
                    cosFac = 0.5_dp*(1.0_dp + cos(pi*dabs(wakeCentersYT_mat(turbI, turb) &
                                     & - turbineYw(turbI))/rmax))

                    if (useaUbU) then
                        wakeEffCoeffPerZone = wakeEffCoeffPerZone + &
                        (((cosFac*rotorDiameter(turb))/(rotorDiameter(turb)+2.0_dp*keArray(turb) &
                        *mmU(zone)*deltax))**2)*wakeOverlapTRel_mat(turbI, turb, zone)
                    else
                        wakeEffCoeffPerZone = wakeEffCoeffPerZone + &
                        (((cosFac*rotorDiameter(turb))/(rotorDiameter(turb)+2.0_dp*keArray(turb) &
                        *MU(zone)*deltax))**2)*wakeOverlapTRel_mat(turbI, turb, zone)
                    end if

                end do
                wakeEffCoeff = wakeEffCoeff + (a(turb)*wakeEffCoeffPerZone)**2
            end if
        end do
        wakeEffCoeff = 1.0_dp - 2.0_dp*sqrt(wakeEffCoeff)
        ! print *, 2.0_dp*sqrt(wakeEffCoeff)

        ! multiply the inflow speed with the wake coefficients to find effective wind
        ! speed at turbine
        wtVelocity(turbI) = wtVelocity(turbI)*wakeEffCoeff
        loss(turbI) = 2.0_dp*sqrt(wakeEffCoeff)
    end do

end subroutine FlorisWake
