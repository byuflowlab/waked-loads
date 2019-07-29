

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


subroutine argsort(array_size,array,sorted_indices)
    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: array_size
    real(dp), dimension(array_size), intent(in) :: array

    ! out
    integer, dimension(array_size), intent(out) :: sorted_indices

    ! local
    integer :: i, j, rank


    sorted_indices = 0.0d0

    ! print *, 'array size ', array_size

    do i = 1, array_size
        rank = 1
        do j = 1, array_size
            if (array(j) < array(i)) then
                rank = rank + 1
            end if
        end do
        ! print *, rank
        sorted_indices(i) = rank
    end do

    ! print *, sorted_indices

end subroutine argsort
