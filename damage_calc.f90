

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



subroutine rotor_amount_waked(dy,wake_radius,rotor_diameter,amnt_waked)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)
    ! in
    real(dp), intent(in) :: dy, wake_radius, rotor_diameter
    ! out
    real(dp), intent(out) :: amnt_waked

    ! local
    real(dp) :: R, a_wake, a_turb, p1, p2, p3, dy_abs
    real(dp), parameter :: pi = 3.141592653589793_dp

    R = wake_radius

    dy_abs = ABS(dy)

    if (dy_abs > (R+(rotor_diameter/2.))) then
        amnt_waked = 0.
    else if ((dy_abs+(rotor_diameter/2.)) < R .AND. (rotor_diameter/2.) < R) then
        amnt_waked = 1.
    else if ((dy_abs+R) < (rotor_diameter/2.) .AND. (rotor_diameter/2.) > R) then
        a_wake = pi*R**2
        a_turb = pi*(rotor_diameter/2.)**2
        amnt_waked = a_wake/a_turb
    else if (R == 0.) then
        amnt_waked = 0.
    else
        p1 = (rotor_diameter/2.)**2*ACOS((dy_abs**2+(rotor_diameter/2.)**2-R**2)/ &
                        & (2.*dy_abs*(rotor_diameter/2.)))
        p2 = R**2*ACOS((dy_abs**2+R**2-(rotor_diameter/2.)**2)/(2.*dy_abs*R))
        p3 = -0.5*SQRT((-dy_abs+(rotor_diameter/2.)+R)*(dy_abs+(rotor_diameter/2.)-R)* &
                        & (dy_abs-(rotor_diameter/2.)+R)*(dy_abs+(rotor_diameter/2.)+R))
        a_turb = pi*(rotor_diameter/2.)**2
        amnt_waked = (p1+p2+p3)/a_turb
    end if

end subroutine rotor_amount_waked



subroutine combine_damage(nTurbines,turbineX,turbineY,turb_index,damage_free,&
                    & damage_close,damage_far,rotor_diameter,wake_radius,damage_out)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines, turb_index
    real(dp), dimension(nTurbines), intent(in) :: turbineX, turbineY
    real(dp), dimension(nTurbines,nTurbines), intent(in) :: wake_radius
    real(dp), intent(in) :: damage_free, damage_close, damage_far, rotor_diameter

    ! out
    real(dp), intent(out) :: damage_out

    ! local
    real(dp) :: unwaked, dy
    integer :: num, ti, waking, k
    real(dp), dimension(nTurbines) :: rotor_waked, dx_dist, indices, waked_array, dx_array, down
    real(dp), parameter :: pi = 3.141592653589793_dp

    ti = turb_index + 1

    do waking = 1, nTurbines
        dx_dist(waking) = turbineX(ti)-turbineX(waking)
        dy = turbineY(ti)-turbineY(waking)
        call rotor_amount_waked(dy,wake_radius(ti,waking),rotor_diameter,rotor_waked(waking))
    end do

    num = 1
    call argsort(nTurbines,dx_dist,indices)

    damage_out = 0.

    waked_array = 0.0d0
    dx_array = 0.0d0

    do waking = 1, nTurbines
        if (dx_dist(indices(waking)) > 0. .AND. rotor_waked(indices(waking)) > SUM(waked_array(1:num))) then
            waked_array(num) = rotor_waked(indices(waking))-SUM(waked_array(1:num))
            dx_array(num) = dx_dist(indices(waking))
            num = num + 1
        end if
    end do

    down = dx_array/rotor_diameter
    unwaked = 1.-SUM(waked_array)

    do k = 1, nTurbines
        if (down(k) .NE. 0.) then
            if (down(k) < 4.) then
                  damage_out = damage_out + damage_close*waked_array(k)
            else if (down(k) > 10.) then
                  damage_out = damage_out + damage_far*waked_array(k)
            else
                  damage_out = damage_out + ((damage_close*(10.-down(k))/6.)+(damage_far*(down(k)-4.)/6.))*waked_array(k)
            end if
        end if
    end do

    damage_out = damage_out + damage_free*unwaked

end subroutine combine_damage



subroutine argsort(array_size,array,sorted_indices)
    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: array_size
    real(dp), dimension(array_size), intent(in) :: array

    ! out
    real(dp), dimension(array_size), intent(out) :: sorted_indices

    ! local
    integer :: i, j, rank


    sorted_indices = 0.0d0

    do i = 1, array_size
        rank = 1
        do j = 1, array_size
            if (array(j) < array(i)) then
                rank = rank + 1
            end if
        end do
        sorted_indices(i) = rank
    end do

end subroutine argsort
