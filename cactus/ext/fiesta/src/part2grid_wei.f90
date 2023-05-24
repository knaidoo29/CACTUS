
subroutine weight_cic(xp, xg, dl, w)

  ! Cloud-in-Cell weigths in 1D.
  !
  ! Parameters
  ! ----------
  ! xp : float
  !   Coordinate of the particle.
  ! xg : float
  !   Coordinate of the grid point.
  ! dl : float
  !   Size of the cell.
  !
  ! Returns
  ! -------
  ! w : float
  !   Weight

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! define variales.

  real(kind=dp), intent(in) :: xp, xg, dl
  real(kind=dp), intent(out) :: w

  real(kind=dp) :: dx

  ! absolute difference between particle and grid point.

  dx = abs(xp - xg)

  ! calculating weight

  if (dx .LE. dl) then
    w = (1. - (dx/dl))/dl
  else
    w = 0.
  end if

end subroutine weight_cic

subroutine weight_tsc(xp, xg, dl, w)

  ! Triangular-shaped-cloud weigths in 1D.
  !
  ! Parameters
  ! ----------
  ! xp : float
  !   Coordinate of the particle.
  ! xg : float
  !   Coordinate of the grid point.
  ! dl : float
  !   Size of the cell.
  !
  ! Returns
  ! -------
  ! w : float
  !   Weight

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! define variables

  real(kind=dp), intent(in) :: xp, xg, dl
  real(kind=dp), intent(out) :: w

  real(kind=dp) :: dx

  ! absolute difference between particle and grid point.

  dx = abs(xp - xg)

  ! calculating weight

  if (dx .LE. dl/2.) then
    w = 3./4. - (dx/dl)**2.
  else if (dx .LE. 3.*dl/2.) then
    w = (1./2.)*((3./2. - (dx/dl))**2.)
  else
    w = 0.
  end if

  w = w / dl

end subroutine weight_tsc
