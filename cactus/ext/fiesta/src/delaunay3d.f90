include "matrix.f90"
include "polyhedron.f90"


subroutine delaunay_volume_3d(x, y, z, del_vert0, del_vert1, del_vert2, del_vert3, npart, nvert, volumes)

  ! Determines the volume for each simplex.
  !
  ! Parameters
  ! ----------
  ! x : array
  !   X-coordinates.
  ! y : array
  !   Y-coordinates.
  ! z : array
  !   Z-coordinates.
  ! del_vert0 : array
  !   Index for vertex 0 of each simplices.
  ! del_vert1 : array
  !   Index for vertex 1 of each simplices.
  ! del_vert2 : array
  !   Index for vertex 2 of each simplices.
  ! del_vert3 : array
  !   Index for vertex 3 of each simplices.
  ! npart : int
  !   Number of points.
  ! nvert : int
  !   Number of vertices.
  !
  ! Returns
  ! -------
  ! volumes : array
  !   The volumes of each simplex.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! Declare variables.

  integer, intent(in) :: npart, nvert
  real(kind=dp), intent(in) :: x(npart), y(npart), z(npart)
  integer, intent(in) :: del_vert0(nvert), del_vert1(nvert), del_vert2(nvert), del_vert3(nvert)
  real(kind=dp), intent(out) :: volumes(nvert)

  integer :: i, i0, i1, i2, i3
  real(kind=dp) :: simplex_vol

  do i = 1, nvert

    i0 = del_vert0(i) + 1
    i1 = del_vert1(i) + 1
    i2 = del_vert2(i) + 1
    i3 = del_vert3(i) + 1

    call tetrahedron_volume(x(i0), y(i0), z(i0), x(i1), y(i1), z(i1), x(i2), y(i2), z(i2), x(i3), y(i3), z(i3), simplex_vol)

    volumes(i) = simplex_vol

  end do

end subroutine delaunay_volume_3d


subroutine sum_delaunay_vol_4_points_3d(delaunay_vol, del_vert0, del_vert1, del_vert2, del_vert3, npart, nvert, point_vol)

    ! Finds the Delaunay volume for each point.
    !
    ! Parameters
    ! ----------
    ! delaunay_vol : array
    !   Delaunay volumes.
    ! del_vert0 : array
    !   Index for vertex 0 of each simplices.
    ! del_vert1 : array
    !   Index for vertex 1 of each simplices.
    ! del_vert2 : array
    !   Index for vertex 2 of each simplices.
    ! npart : int
    !   Number of points.
    ! nvert : int
    !   Number of vertices.
    !
    ! Returns
    ! -------
    ! point_vol : array
    !   Sum of delaunay value for each point.

    implicit none
    integer, parameter :: dp = kind(1.d0)

    ! Declare variables.

    integer, intent(in) :: npart, nvert
    real(kind=dp), intent(in) :: delaunay_vol(nvert)
    integer, intent(in) :: del_vert0(nvert), del_vert1(nvert), del_vert2(nvert), del_vert3(nvert)
    real(kind=dp), intent(out) :: point_vol(npart)

    integer :: i, i0, i1, i2, i3

    do i = 1, npart
      point_vol(i) = 0.
    end do

    do i = 1, nvert
      i0 = del_vert0(i) + 1
      i1 = del_vert1(i) + 1
      i2 = del_vert2(i) + 1
      i3 = del_vert3(i) + 1
      point_vol(i0) = point_vol(i0) + delaunay_vol(i)/4.
      point_vol(i1) = point_vol(i1) + delaunay_vol(i)/4.
      point_vol(i2) = point_vol(i2) + delaunay_vol(i)/4.
      point_vol(i3) = point_vol(i3) + delaunay_vol(i)/4.
    end do

end subroutine sum_delaunay_vol_4_points_3d


subroutine get_delf0_3d(x, y, z, f, del_vert0, del_vert1, del_vert2, del_vert3, npart, nvert, delf0)

  ! Determines delf0 for each simplices.
  !
  ! Parameters
  ! ----------
  ! x : array
  !   X-coordinates.
  ! y : array
  !   Y-coordinates.
  ! z : array
  !   Z-coordinates.
  ! f : array
  !   Field values at these points.
  ! del_vert0 : array
  !   Index for vertex 0 of each simplices.
  ! del_vert1 : array
  !   Index for vertex 1 of each simplices.
  ! del_vert2 : array
  !   Index for vertex 2 of each simplices.
  ! del_vert3 : array
  !   Index for vertex 3 of each simplices.
  ! npart : int
  !   Number of points.
  ! nvert : int
  !   Number of vertices.
  !
  ! Returns
  ! -------
  ! delf0 : array
  !   The 3D difference in each simplices.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! Declare variables.

  integer, intent(in) :: npart, nvert
  real(kind=dp), intent(in) :: x(npart), y(npart), z(npart), f(npart)
  integer, intent(in) :: del_vert0(nvert), del_vert1(nvert), del_vert2(nvert),  del_vert3(nvert)
  real(kind=dp), intent(out) :: delf0(3*nvert)

  integer :: i, i0, i1, i2, i3
  real(kind=dp) :: dx1, dx2, dx3, dy1, dy2, dy3, dz1, dz2, dz3, df1, df2, df3, m(9), invm(9)

  do i = 1, nvert

    i0 = del_vert0(i) + 1
    i1 = del_vert1(i) + 1
    i2 = del_vert2(i) + 1
    i3 = del_vert3(i) + 1

    dx1 = x(i1) - x(i0)
    dx2 = x(i2) - x(i0)
    dx3 = x(i3) - x(i0)

    dy1 = y(i1) - y(i0)
    dy2 = y(i2) - y(i0)
    dy3 = y(i3) - y(i0)

    dz1 = z(i1) - z(i0)
    dz2 = z(i2) - z(i0)
    dz3 = z(i3) - z(i0)

    df1 = f(i1) - f(i0)
    df2 = f(i2) - f(i0)
    df3 = f(i3) - f(i0)

    m(1) = dx1
    m(2) = dy1
    m(3) = dz1

    m(4) = dx2
    m(5) = dy2
    m(6) = dz2

    m(7) = dx3
    m(8) = dy3
    m(9) = dz3

    call inv3by3(m, invm)

    delf0(3*i-2) = invm(1)*df1 + invm(2)*df2 + invm(3)*df3
    delf0(3*i-1) = invm(4)*df1 + invm(5)*df2 + invm(6)*df3
    delf0(3*i)   = invm(7)*df1 + invm(8)*df2 + invm(9)*df3

  end do

end subroutine get_delf0_3d


subroutine delaunay_estimate_3d(simplices, x, y, z, x0, y0, z0, f0, delf0, npart, nsimp0, f_est)

  ! Estimates a field from Delaunay tesselation.
  !
  ! Parameters
  ! ----------
  ! x : array
  !   X-coordinates for estimates.
  ! y : array
  !   Y-coordinates for estimates.
  ! z : array
  !   Z-coordinates for estimates.
  ! x0 : array
  !   X-coordinate of vertex 0 of each simplices.
  ! y0 : array
  !   Y-coordinate of vertex 0 of each simplices.
  ! z0 : array
  !   Z-coordinate of vertex 0 of each simplices.
  ! f0 : array
  !   Field values at vertex 0 of each simplices.
  ! delf0 : array
  !   The 3D difference in each simplices.
  ! npart : int
  !   Number of coordinates for estimates.
  ! nsimp0 : int
  !   Number of simplices.
  !
  ! Returns
  ! -------
  ! f_est : array
  !   Estimates of the field.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! Declare variables.

  integer, intent(in) :: npart, nsimp0
  integer, intent(in) :: simplices(npart)
  real(kind=dp), intent(in) :: x(npart), y(npart), z(npart)
  real(kind=dp), intent(in) :: x0(nsimp0), y0(nsimp0), z0(nsimp0), f0(nsimp0), delf0(3*nsimp0)
  real(kind=dp), intent(out) :: f_est(npart)

  integer :: i, j

  do i = 1, npart
    j = simplices(i) + 1
    f_est(i) = f0(j) + delf0(3*j-2)*(x(i) - x0(j)) + delf0(3*j-1)*(y(i) - y0(j)) + delf0(3*j)*(z(i) - z0(j))
  end do

end subroutine delaunay_estimate_3d
