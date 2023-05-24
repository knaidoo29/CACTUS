
subroutine which_pix_id_scalar(x, xmin, dx, pix_id)

  ! Find pixel along a defined grid.
  !
  ! Parameters
  ! ----------
  ! x : float
  !   X-coordinates.
  ! xmin : float
  !   Minimum along the grid.
  ! dx : float
  !   pixel width.
  !
  ! Returns
  ! -------
  ! pix_id : int
  !   The pixel the points are located in.

  implicit none

  ! Parameter declarations

  integer, parameter :: dp = kind(1.d0)

  real(kind=dp), intent(in) :: x, xmin, dx
  integer, intent(out) :: pix_id

  ! Main

  pix_id = int(floor((x-xmin)/dx))

end subroutine which_pix_id_scalar


subroutine which_pix_id_array(x, xmin, dx, xlen, pix_id)

  ! Find pixel along a defined grid.
  !
  ! Parameters
  ! ----------
  ! x : array
  !   X-coordinates.
  ! xmin : float
  !   Minimum along the grid.
  ! dx : float
  !   pixel width.
  !
  ! Returns
  ! -------
  ! pix_id : int
  !   The pixel the points are located in.

  implicit none

  ! Parameter declarations

  integer, parameter :: dp = kind(1.d0)
  integer, intent(in) :: xlen
  real(kind=dp), intent(in) :: x(xlen), xmin, dx
  integer, intent(out) :: pix_id(xlen)
  integer :: i

  ! Main

  do i = 1, xlen
    call which_pix_id_scalar(x(i), xmin, dx, pix_id(i))
  end do

end subroutine which_pix_id_array
