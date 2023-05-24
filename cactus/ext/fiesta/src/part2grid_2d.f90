include "part2grid_pix.f90"
include "part2grid_wei.f90"


subroutine part2grid_ngp_2d(x, y, f, xlength, ylength, xmin, ymin, npart &
  , nxgrid, nygrid, fgrid)

  ! Nearest-grid-point assignment in 2D.
  !
  ! Parameters
  ! ----------
  ! x, y : array
  !   Cartesian coordinate system.
  ! f : array
  !   Field values are x & y coordinates.
  ! xlength, ylength : float
  !   Length of the box along the x and y coordinates.
  ! xmin, ymin : float
  !   Minimum values along the x and y axis.
  ! npart : int
  !   Number of x and y coordinates.
  ! nxgrid, nygrid : int
  !   Number of grids along x and y coordinates.
  ! fgrid : array
  !   NGP field assignments.

  implicit none

  ! Parameter declarations

  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: npart, nxgrid, nygrid
  real(kind=dp), intent(in) :: x(npart), y(npart), f(npart)
  real(kind=dp), intent(in) :: xlength, ylength, xmin, ymin
  real(kind=dp), intent(out) :: fgrid(nxgrid*nygrid)

  integer :: i, xpix, ypix, pix
  real(kind=dp) :: wngp, dx, dy, xp, yp, fp

  ! Main

  dx = xlength / real(nxgrid)
  dy = ylength / real(nygrid)

  wngp = 1./(dx*dy)

  do i = 1, nxgrid*nygrid
    fgrid(i) = 0.
  end do

  do i = 1, npart

    xp = x(i)
    yp = y(i)
    fp = f(i)

    call ngp_pix(xp, dx, xmin, xpix)
    call ngp_pix(yp, dy, ymin, ypix)

    if ((xpix .GE. 0) .AND. (xpix .LT. nxgrid) .AND. &
      (ypix .GE. 0) .AND. (ypix .LT. nygrid)) then

      call pix1dto2d_scalar(xpix, ypix, nygrid, pix)

      fgrid(pix+1) = fgrid(pix+1) + fp*wngp

    end if

  end do

end subroutine part2grid_ngp_2d


subroutine part2grid_cic_2d(x, y, f, xlength, ylength, xmin, ymin, npart &
  , nxgrid, nygrid, periodx, periody, fgrid)

  ! Cloud-in-cell assignment in 2D.
  !
  ! Parameters
  ! ----------
  ! x, y : array
  !   Cartesian coordinate system.
  ! f : array
  !   Field values are x & y coordinates.
  ! xlength, ylength : float
  !   Length of the box along the x and y coordinates.
  ! xmin, ymin : float
  !   Minimum values along the x and y axis.
  ! npart : int
  !   Number of x and y coordinates.
  ! nxgrid, nygrid : int
  !   Number of grids along x and y coordinates.
  ! periodx, periody, periodz : bool
  !   Periodic boundary conditions.
  ! fgrid : array
  !   CIC field assignments.

  implicit none

  ! Parameter declarations

  integer, parameter :: dp = kind(1.d0)
  integer, intent(in) :: npart, nxgrid, nygrid
  logical, intent(in) :: periodx, periody
  real(kind=dp), intent(in) :: x(npart), y(npart), f(npart)
  real(kind=dp), intent(in) :: xlength, ylength, xmin, ymin
  real(kind=dp), intent(out) :: fgrid(nxgrid*nygrid)
  integer :: i, j1, j2, xpix(2), ypix(2), pix
  real(kind=dp) :: wx, wy, dx, dy, xp, yp, xg(2), yg(2), fp

  ! Main

  dx = xlength / real(nxgrid)
  dy = ylength / real(nygrid)

  do i = 1, nxgrid*nygrid
    fgrid(i) = 0.
  end do

  do i = 1, npart

    xp = x(i)
    yp = y(i)
    fp = f(i)

    call cic_pix(xp, dx, xmin, xpix)
    call cic_pix(yp, dx, ymin, ypix)
    call xgrids(xpix, 2, dx, xmin, xg)
    call xgrids(ypix, 2, dx, ymin, yg)

    if (periodx .EQV. .TRUE.) then
      call periodic_pix(xpix, 2, nxgrid)
    end if

    if (periody .EQV. .TRUE.) then
      call periodic_pix(ypix, 2, nygrid)
    end if

    do j1 = 1, 2
      do j2 = 1, 2
        if ((xpix(j1) .GE. 0) .AND. (xpix(j1) .LT. nxgrid) &
          .AND. (ypix(j2) .GE. 0) .AND. (ypix(j2) .LT. nygrid)) then

          call pix1dto2d_scalar(xpix(j1), ypix(j2), nygrid, pix)
          call weight_cic(xp, xg(j1), dx, wx)
          call weight_cic(yp, yg(j2), dy, wy)

          fgrid(pix+1) = fgrid(pix+1) + fp*wx*wy

        end if
      end do
    end do

  end do

end subroutine part2grid_cic_2d


subroutine part2grid_tsc_2d(x, y, f, xlength, ylength, xmin, ymin, npart &
  , nxgrid, nygrid, periodx, periody, fgrid)

  ! Triangular-shaped-cloud assignment in 2D.
  !
  ! Parameters
  ! ----------
  ! x, y : array
  !   Cartesian coordinate system.
  ! f : array
  !   Field values are x & y coordinates.
  ! xlength, ylength : float
  !   Length of the box along the x and y coordinates.
  ! xmin, ymin : float
  !   Minimum values along the x and y axis.
  ! npart : int
  !   Number of x and y coordinates.
  ! nxgrid, nygrid : int
  !   Number of grids along x and y coordinates.
  ! periodx, periody : bool
  !   Periodic boundary conditions.
  ! fgrid : array
  !   TSC field assignments.

  implicit none

  ! Parameter declarations

  integer, parameter :: dp = kind(1.d0)
  integer, intent(in) :: npart, nxgrid, nygrid
  logical, intent(in) :: periodx, periody
  real(kind=dp), intent(in) :: x(npart), y(npart), f(npart)
  real(kind=dp), intent(in) :: xlength, ylength, xmin, ymin
  real(kind=dp), intent(out) :: fgrid(nxgrid*nygrid)
  integer :: i, j1, j2, xpix(3), ypix(3), pix
  real(kind=dp) :: wx, wy, dx, dy, xp, yp, xg(3), yg(3), fp

  ! Main

  dx = xlength / real(nxgrid)
  dy = ylength / real(nygrid)

  do i = 1, nxgrid*nygrid
    fgrid(i) = 0.
  end do

  do i = 1, npart

    xp = x(i)
    yp = y(i)
    fp = f(i)

    call tsc_pix(xp, dx, xmin, xpix)
    call tsc_pix(yp, dy, ymin, ypix)
    call xgrids(xpix, 3, dx, xmin, xg)
    call xgrids(ypix, 3, dy, ymin, yg)

    if (periodx .EQV. .TRUE.) then
      call periodic_pix(xpix, 3, nxgrid)
    end if

    if (periody .EQV. .TRUE.) then
      call periodic_pix(ypix, 3, nygrid)
    end if

    do j1 = 1, 3
      do j2 = 1, 3
        if ((xpix(j1) .GE. 0) .AND. (xpix(j1) .LT. nxgrid) &
          .AND. (ypix(j2) .GE. 0) .AND. (ypix(j2) .LT. nygrid)) then

          call pix1dto2d_scalar(xpix(j1), ypix(j2), nygrid, pix)
          call weight_tsc(xp, xg(j1), dx, wx)
          call weight_tsc(yp, yg(j2), dy, wy)

          fgrid(pix+1) = fgrid(pix+1) + fp*wx*wy

        end if
      end do
    end do

  end do

end subroutine part2grid_tsc_2d
