include "grid.f90"


subroutine which_pix(x, dx, xmin, pix)

  ! Find pixel along a defined grid.
  !
  ! Parameters
  ! ----------
  ! xmin : float
  !   Minimum along the grid.
  ! dx : float
  !   pixel width.
  ! x : float
  !   A point for which we would like to determine
  !
  ! Returns
  ! -------
  ! pix : int
  !   The pixel the point corresponds to.

  implicit none

  ! Parameter declarations

  integer, parameter :: dp = kind(1.d0)

  real(kind=dp), intent(in) :: xmin, dx, x
  integer, intent(out) :: pix

  ! Main

  pix = int(floor((x-xmin)/dx))

end subroutine which_pix


subroutine which_pixs(x, dx, xmin, npix, pixs)

  ! Find pixel along a defined grid.
  !
  ! Parameters
  ! ----------
  ! xmin : float
  !   Minimum along the grid.
  ! dx : float
  !   pixel width.
  ! x : float
  !   A point for which we would like to determine
  !
  !
  ! Returns
  ! -------
  ! pix : int
  !   The pixel the point corresponds to.

  implicit none

  ! Parameter declarations

  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: npix
  real(kind=dp), intent(in) :: xmin, dx, x(npix)
  integer, intent(out) :: pixs(npix)

  integer :: i

  ! Main

  do i=1, npix
    call which_pix(x(i), dx, xmin, pixs(i))
  end do

end subroutine which_pixs


subroutine pix1dto2d(xpix, ypix, xlen, ylen, ygrid, pix)

  ! Maps pixels given along a single axis in x and y onto a 2d grid flattened.
  !
  ! Parameters
  ! ----------
  ! xpix : int
  !   Pixel indices along the x axis grid.
  ! ypix : int
  !   Pixel indices along the y axis grid.
  ! xlen : int
  !   Size of the xpix.
  ! ylen : int
  !   Size of the ypix.
  ! ygrid : int
  !   Length of y axis grid.
  !
  ! Returns
  ! -------
  ! pix : int
  !   2d grid pixel.

  implicit none

  integer, intent(in) :: xlen, ylen, ygrid
  integer, intent(in) :: xpix(xlen), ypix(ylen)
  integer, intent(out) :: pix(xlen*ylen)

  integer :: i, j, ii

  ! Main

  ii = 1

  do i = 1, xlen
    do j = 1, ylen
      if ((xpix(i) .NE. -1) .AND. (ypix(j) .NE. -1)) then
        pix(ii) = ypix(j) + ygrid*xpix(i)
      else
        pix(ii) = -1
      end if
      ii = ii + 1
    end do
  end do

end subroutine pix1dto2d


subroutine pix1dto3d(xpix, ypix, zpix, xlen, ylen, zlen, ygrid, zgrid, pix)

  ! Maps pixels given along a single axis in x, y and z onto a 3d grid
  ! flattened.
  !
  ! Parameters
  ! ----------
  ! xpix : int
  !   Pixel indices along the x axis grid.
  ! ypix : int
  !   Pixel indices along the y axis grid.
  ! xlen : int
  !   Size of the xpix.
  ! ylen : int
  !   Size of the ypix.
  ! zlen : int
  !   Size of the zpix.
  ! ygrid : int
  !   Length of y axis grid.
  ! zgrid : int
  !   Length of z axis grid.
  !
  ! Returns
  ! -------
  ! pix : int
  !   2d grid pixel.

  implicit none

  integer, intent(in) :: xlen, ylen, zlen, ygrid, zgrid
  integer, intent(in) :: xpix(xlen), ypix(ylen), zpix(zlen)
  integer, intent(out) :: pix(xlen*ylen*zlen)

  integer :: i, j, k, ii

  ! Main

  ii = 1

  do i = 1, xlen
    do j = 1, ylen
      do k = 1, zlen
        if ((xpix(i) .NE. -1) .AND. (ypix(j) .NE. -1) .AND. &
          (zpix(k) .NE. -1)) then
          pix(ii) = zpix(k) + zgrid*(ypix(j) + ygrid*xpix(i))
        else
          pix(ii) = -1
        end if
        ii = ii + 1
      end do
    end do
  end do

end subroutine pix1dto3d



subroutine pix1dto2d_scalar(xpix, ypix, ygrid, pix)

  ! Maps pixels given along a single axis in x and y onto a 2d grid flattened.
  !
  ! Parameters
  ! ----------
  ! xpix : int
  !   Pixel indices along the x axis grid.
  ! ypix : int
  !   Pixel indices along the y axis grid.
  ! ygrid : int
  !   Length of y axis grid.
  !
  ! Returns
  ! -------
  ! pix : int
  !   2d grid pixel.

  implicit none

  integer, intent(in) :: xpix, ypix, ygrid
  integer, intent(out) :: pix

  ! Main

  if ((xpix .NE. -1) .AND. (ypix .NE. -1)) then
    pix = ypix + ygrid*xpix
  else
    pix = -1
  end if


end subroutine pix1dto2d_scalar


subroutine pix1dto3d_scalar(xpix, ypix, zpix, ygrid, zgrid, pix)

  ! Maps pixels given along a single axis in x, y and z onto a 3d grid
  ! flattened.
  !
  ! Parameters
  ! ----------
  ! xpix : int
  !   Pixel indices along the x axis grid.
  ! ypix : int
  !   Pixel indices along the y axis grid.
  ! ygrid : int
  !   Length of y axis grid.
  ! zgrid : int
  !   Length of z axis grid.
  !
  ! Returns
  ! -------
  ! pix : int
  !   2d grid pixel.

  implicit none

  integer, intent(in) :: ygrid, zgrid
  integer, intent(in) :: xpix, ypix, zpix
  integer, intent(out) :: pix

  ! Main

  if ((xpix .NE. -1) .AND. (ypix .NE. -1) .AND. (zpix .NE. -1)) then
    pix = zpix + zgrid*(ypix + ygrid*xpix)
  else
    pix = -1
  end if

end subroutine pix1dto3d_scalar


subroutine periodic_pix(pix, pixlen, ngrid)

  ! Applies periodic condition to pixel index.
  !
  ! Parameters
  ! ----------
  ! pix : array(int)
  !   Pixel index array.
  ! pixlen : int
  !   Length pixel index array.
  ! ngrid : int
  !   Grid dimensions.
  !
  ! Returns
  ! -------
  ! pix : array(int)
  !   Periodic pixel index array

  implicit none

  ! Parameter declarations

  integer, parameter :: dp = kind(1.d0)
  integer, intent(in) :: pixlen, ngrid
  integer, intent(inout) :: pix(pixlen)
  integer :: i

  ! Main

  do i = 1, pixlen
    if (pix(i) .LT. 0) then
      pix(i) = pix(i) + ngrid
    else if (pix(i) .GE. ngrid) then
      pix(i) = pix(i) - ngrid
    end if
  end do

end subroutine periodic_pix

subroutine ngp_pix(x, dx, xmin, pix)

  ! Nearest-grid-point pixel index.
  !
  ! Parameters
  ! ----------
  ! x : array(float)
  !   Array of x-coordinates.
  ! dx : float
  !   Grid size.
  ! xmin : float
  !   Minimum along the grid.
  !
  ! Returns
  ! -------
  ! pix : array(int)
  !   Nearest-grid-point pixel index array.

  implicit none

  ! Parameter declarations

  integer, parameter :: dp = kind(1.d0)
  real(kind=dp), intent(in) :: x, dx, xmin
  integer, intent(out) :: pix

  ! Main

  call which_pix(x, dx, xmin, pix)

end subroutine ngp_pix


subroutine cic_pix(x, dx, xmin, pix)

  ! Cloud-in-cell pixel index.
  !
  ! Parameters
  ! ----------
  ! x : array(float)
  !   Array of x-coordinates.
  ! dx : float
  !   Grid size.
  ! xmin : float
  !   Minimum along the grid.
  !
  ! Returns
  ! -------
  ! pix : array(int)
  !   Cloud-in-cell pixel index array.

  implicit none

  ! Parameter declarations

  integer, parameter :: dp = kind(1.d0)
  real(kind=dp), intent(in) :: x, dx, xmin
  integer, intent(out) :: pix(2)
  integer :: xpix
  real(kind=dp) :: xg

  ! Main

  call which_pix(x, dx, xmin, xpix)
  call xgrid(xpix, dx, xmin, xg)

  if (x .LT. xg) then
    pix(1) = xpix - 1
    pix(2) = xpix
  else
    pix(1) = xpix
    pix(2) = xpix + 1
  end if

end subroutine cic_pix


subroutine tsc_pix(x, dx, xmin, pix)

  ! Triangular-shaped-cloud pixel index.
  !
  ! Parameters
  ! ----------
  ! x : array(float)
  !   Array of x-coordinates.
  ! dx : float
  !   Grid size.
  ! xmin : float
  !   Minimum along the grid.
  !
  ! Returns
  ! -------
  ! pix : array(int)
  !   Triangular-shaped-cloud pixel index array.

  implicit none

  ! Parameter declarations

  integer, parameter :: dp = kind(1.d0)
  real(kind=dp), intent(in) :: x, dx, xmin
  integer, intent(out) :: pix(3)
  integer :: xpix

  ! Main

  call which_pix(x, dx, xmin, xpix)

  pix(1) = xpix - 1
  pix(2) = xpix
  pix(3) = xpix + 1

end subroutine tsc_pix
