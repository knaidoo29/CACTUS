include "part2grid_pix.f90"
include "part2grid_wei.f90"


subroutine part2grid_ngp_3d(x, y, z, f, xlength, ylength, zlength, xmin, ymin &
  , zmin, npart, nxgrid, nygrid, nzgrid, fgrid)

  ! Nearest-grid-point assignment in 3D.
  !
  ! Parameters
  ! ----------
  ! x, y, z : array
  !   Cartesian coordinate system.
  ! f : array
  !   Field values are x, y and z coordinates.
  ! xlength, ylength, zlength : float
  !   Length of the box along the x, y and z coordinates.
  ! xmin, ymin, zmin : float
  !   Minimum values along the x, y and z axis.
  ! npart : int
  !   Number of x, y and z coordinates.
  ! nxgrid, nygrid, nzgrid : int
  !   Number of grids along x, y and z coordinates.
  ! fgrid : array
  !   NGP field assignments.

  implicit none

  ! Parameter declarations

  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: npart, nxgrid, nygrid, nzgrid
  real(kind=dp), intent(in) :: x(npart), y(npart), z(npart), f(npart)
  real(kind=dp), intent(in) :: xlength, ylength, zlength, xmin, ymin, zmin
  real(kind=dp), intent(out) :: fgrid(nxgrid*nygrid*nzgrid)

  integer :: i, xpix, ypix, zpix, pix
  real(kind=dp) :: wngp, dx, dy, dz, xp, yp, zp, fp

  ! Main

  dx = xlength / real(nxgrid)
  dy = ylength / real(nygrid)
  dz = zlength / real(nzgrid)
  wngp = 1./(dx*dy*dz)

  do i = 1, nxgrid*nygrid*nzgrid
    fgrid(i) = 0.
  end do

  do i = 1, npart

    xp = x(i)
    yp = y(i)
    zp = z(i)
    fp = f(i)

    call ngp_pix(xp, dx, xmin, xpix)
    call ngp_pix(yp, dy, ymin, ypix)
    call ngp_pix(zp, dz, zmin, zpix)

    if ((xpix .GE. 0) .AND. (xpix .LT. nxgrid) .AND. (ypix .GE. 0) &
    .AND. (ypix .LT. nygrid) .AND. (zpix .GE. 0) .AND. (zpix .LT. nzgrid)) then

      call pix1dto3d_scalar(xpix, ypix, zpix, nygrid, nzgrid, pix)
      fgrid(pix+1) = fgrid(pix+1) + fp*wngp

    end if

  end do

end subroutine part2grid_ngp_3d


subroutine part2grid_cic_3d(x, y, z, f, xlength, ylength, zlength, xmin, ymin &
  , zmin, npart, nxgrid, nygrid, nzgrid, periodx, periody, periodz, fgrid)

  ! Cloud-in-cell assignment in 2D.
  !
  ! Parameters
  ! ----------
  ! x, y, z : array
  !   Cartesian coordinate system.
  ! f : array
  !   Field values are x, y & z coordinates.
  ! xlength, ylength, zlength : float
  !   Length of the box along the x, y & z coordinates.
  ! xmin, ymin : float
  !   Minimum values along the x, y & z axis.
  ! npart : int
  !   Number of x, y & z coordinates.
  ! nxgrid, nygrid : int
  !   Number of grids along x, y & z coordinates.
  ! periodx, periody, periodz : bool
  !   Periodic boundary conditions.
  ! fgrid : array
  !   CIC field assignments.

  implicit none

  ! Parameter declarations

  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: npart, nxgrid, nygrid, nzgrid
  logical, intent(in) :: periodx, periody, periodz
  real(kind=dp), intent(in) :: x(npart), y(npart), z(npart), f(npart)
  real(kind=dp), intent(in) :: xlength, ylength, zlength, xmin, ymin, zmin
  real(kind=dp), intent(out) :: fgrid(nxgrid*nygrid*nzgrid)

  integer :: i, j1, j2, j3, xpix(2), ypix(2), zpix(2), pix
  real(kind=dp) :: wx, wy, wz, dx, dy, dz, xp, yp, zp, xg(2), yg(2), zg(2), fp

  ! Main

  dx = xlength / real(nxgrid)
  dy = ylength / real(nygrid)
  dz = zlength / real(nzgrid)

  do i = 1, nxgrid*nygrid*nzgrid
    fgrid(i) = 0.
  end do

  do i = 1, npart

    xp = x(i)
    yp = y(i)
    zp = z(i)
    fp = f(i)

    call cic_pix(xp, dx, xmin, xpix)
    call cic_pix(yp, dx, ymin, ypix)
    call cic_pix(zp, dx, zmin, zpix)
    call xgrids(xpix, 2, dx, xmin, xg)
    call xgrids(ypix, 2, dx, ymin, yg)
    call xgrids(zpix, 2, dx, zmin, zg)

    if (periodx .EQV. .TRUE.) then
      call periodic_pix(xpix, 2, nxgrid)
    end if

    if (periody .EQV. .TRUE.) then
      call periodic_pix(ypix, 2, nygrid)
    end if

    if (periodz .EQV. .TRUE.) then
      call periodic_pix(zpix, 2, nzgrid)
    end if

    do j1 = 1, 2
      do j2 = 1, 2
        do j3 = 1, 2
          if ((xpix(j1) .GE. 0) .AND. (xpix(j1) .LT. nxgrid) &
            .AND. (ypix(j2) .GE. 0) .AND. (ypix(j2) .LT. nygrid) &
            .AND. (zpix(j3) .GE. 0) .AND. (zpix(j3) .LT. nzgrid)) then

            call pix1dto3d_scalar(xpix(j1), ypix(j2), zpix(j3), nygrid, nzgrid, pix)
            call weight_cic(xp, xg(j1), dx, wx)
            call weight_cic(yp, yg(j2), dx, wy)
            call weight_cic(zp, zg(j3), dx, wz)

            fgrid(pix+1) = fgrid(pix+1) + fp*wx*wy*wz

          end if
        end do
      end do
    end do

  end do

end subroutine part2grid_cic_3d


subroutine part2grid_tsc_3d(x, y, z, f, xlength, ylength, zlength, xmin, ymin &
  , zmin, npart, nxgrid, nygrid, nzgrid, periodx, periody, periodz, fgrid)

  ! Triangular-shaped-cloud assignment in 3D.
  !
  ! Parameters
  ! ----------
  ! x, y, z : array
  !   Cartesian coordinate system.
  ! f : array
  !   Field values are x, y & z coordinates.
  ! xlength, ylength, zlength : float
  !   Length of the box along the x, y & z coordinates.
  ! xmin, ymin, zmin : float
  !   Minimum values along the x, y & z axis.
  ! npart : int
  !   Number of x, y & z coordinates.
  ! nxgrid, nygrid, nzgrid : int
  !   Number of grids along x, y & z coordinates.
  ! periodx, periody, periodz : bool
  !   Periodic boundary conditions.
  ! fgrid : array
  !   TSC field assignments.

  implicit none

  ! Parameter declarations

  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: npart, nxgrid, nygrid, nzgrid
  logical, intent(in) :: periodx, periody, periodz
  real(kind=dp), intent(in) :: x(npart), y(npart), z(npart), f(npart)
  real(kind=dp), intent(in) :: xlength, ylength, zlength, xmin, ymin, zmin
  real(kind=dp), intent(out) :: fgrid(nxgrid*nygrid*nzgrid)

  integer :: i, j1, j2, j3, xpix(3), ypix(3), zpix(3), pix
  real(kind=dp) :: wx, wy, wz, dx, dy, dz, xp, yp, zp, xg(3), yg(3), zg(3), fp

  dx = xlength / real(nxgrid)
  dy = ylength / real(nygrid)
  dz = zlength / real(nzgrid)

  do i = 1, nxgrid*nygrid*nzgrid
    fgrid(i) = 0.
  end do

  do i = 1, npart

    xp = x(i)
    yp = y(i)
    zp = z(i)
    fp = f(i)

    call tsc_pix(xp, dx, xmin, xpix)
    call tsc_pix(yp, dy, ymin, ypix)
    call tsc_pix(zp, dz, zmin, zpix)
    call xgrids(xpix, 3, dx, xmin, xg)
    call xgrids(ypix, 3, dy, ymin, yg)
    call xgrids(zpix, 3, dz, zmin, zg)

    if (periodx .EQV. .TRUE.) then
      call periodic_pix(xpix, 3, nxgrid)
    end if

    if (periody .EQV. .TRUE.) then
      call periodic_pix(ypix, 3, nygrid)
    end if

    if (periodz .EQV. .TRUE.) then
      call periodic_pix(zpix, 3, nzgrid)
    end if

    do j1 = 1, 3
      do j2 = 1, 3
        do j3 = 1, 3
          if ((xpix(j1) .GE. 0) .AND. (xpix(j1) .LT. nxgrid) &
            .AND. (ypix(j2) .GE. 0) .AND. (ypix(j2) .LT. nygrid) &
            .AND. (zpix(j3) .GE. 0) .AND. (zpix(j3) .LT. nzgrid)) then

            call pix1dto3d_scalar(xpix(j1), ypix(j2), zpix(j3), nygrid, nzgrid, pix)
            call weight_tsc(xp, xg(j1), dx, wx)
            call weight_tsc(yp, yg(j2), dx, wy)
            call weight_tsc(zp, zg(j3), dx, wz)

            fgrid(pix+1) = fgrid(pix+1) + fp*wx*wy*wz

          end if
        end do
      end do
    end do

  end do

end subroutine part2grid_tsc_3d
