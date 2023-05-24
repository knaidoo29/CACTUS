include "matrix.f90"
include "polygon.f90"


subroutine delaunay_area_2d(x, y, del_vert0, del_vert1, del_vert2, npart, nvert, areas)

  ! Determines area for each simplex.
  !
  ! Parameters
  ! ----------
  ! x : array
  !   X-coordinates.
  ! y : array
  !   Y-coordinates.
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
  ! areas : array
  !   The areas of each simplex.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! Declare variables.

  integer, intent(in) :: npart, nvert
  real(kind=dp), intent(in) :: x(npart), y(npart)
  integer, intent(in) :: del_vert0(nvert), del_vert1(nvert), del_vert2(nvert)
  real(kind=dp), intent(out) :: areas(nvert)

  integer :: i, i0, i1, i2
  real(kind=dp) :: simplex_area

  do i = 1, nvert

    i0 = del_vert0(i) + 1
    i1 = del_vert1(i) + 1
    i2 = del_vert2(i) + 1

    call triangle_area(x(i0), y(i0), x(i1), y(i1), x(i2), y(i2), simplex_area)

    areas(i) = simplex_area

  end do

end subroutine delaunay_area_2d


subroutine sum_delaunay4points_2d(delaunay_value, del_vert0, del_vert1, del_vert2, npart, nvert, point_del_sum)

    ! Sums the values of each Delaunay tesselation for each point.
    !
    ! Parameters
    ! ----------
    ! delaunay_value : array
    !   Delaunay values.
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
    ! point_del_sum : array
    !   Sum of delaunay value for each point.

    implicit none
    integer, parameter :: dp = kind(1.d0)

    ! Declare variables.

    integer, intent(in) :: npart, nvert
    real(kind=dp), intent(in) :: delaunay_value(nvert)
    integer, intent(in) :: del_vert0(nvert), del_vert1(nvert), del_vert2(nvert)
    real(kind=dp), intent(out) :: point_del_sum(npart)

    integer :: i, i0, i1, i2

    do i = 1, npart
      point_del_sum(i) = 0.
    end do

    do i = 1, nvert
      i0 = del_vert0(i) + 1
      i1 = del_vert1(i) + 1
      i2 = del_vert2(i) + 1
      point_del_sum(i0) = point_del_sum(i0) + delaunay_value(i)/3.
      point_del_sum(i1) = point_del_sum(i1) + delaunay_value(i)/3.
      point_del_sum(i2) = point_del_sum(i2) + delaunay_value(i)/3.
    end do

end subroutine sum_delaunay4points_2d


subroutine get_delf0_2d(x, y, f, del_vert0, del_vert1, del_vert2, npart, nvert, delf0)

  ! Determines delf0 for each simplices.
  !
  ! Parameters
  ! ----------
  ! x : array
  !   X-coordinates.
  ! y : array
  !   Y-coordinates.
  ! f : array
  !   Field values at these points.
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
  ! delf0 : array
  !   The 2D difference in each simplices.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! Declare variables.

  integer, intent(in) :: npart, nvert
  real(kind=dp), intent(in) :: x(npart), y(npart), f(npart)
  integer, intent(in) :: del_vert0(nvert), del_vert1(nvert), del_vert2(nvert)
  real(kind=dp), intent(out) :: delf0(2*nvert)

  integer :: i, i0, i1, i2
  real(kind=dp) :: dx1, dx2, dy1, dy2, df1, df2, m(4), invm(4)

  do i = 1, nvert

    i0 = del_vert0(i) + 1
    i1 = del_vert1(i) + 1
    i2 = del_vert2(i) + 1

    dx1 = x(i1) - x(i0)
    dx2 = x(i2) - x(i0)

    dy1 = y(i1) - y(i0)
    dy2 = y(i2) - y(i0)

    df1 = f(i1) - f(i0)
    df2 = f(i2) - f(i0)

    m(1) = dx1
    m(2) = dy1
    m(3) = dx2
    m(4) = dy2

    call inv2by2(m, invm)

    delf0(2*i-1) = invm(1)*df1 + invm(2)*df2
    delf0(2*i) = invm(3)*df1 + invm(4)*df2

  end do

end subroutine get_delf0_2d


subroutine delaunay_estimate_2d(simplices, x, y, x0, y0, f0, delf0, npart, nsimp0, f_est)

  ! Estimates a field from Delaunay tesselation.
  !
  ! Parameters
  ! ----------
  ! x : array
  !   X-coordinates for estimates.
  ! y : array
  !   Y-coordinates for estimates.
  ! x0 : array
  !   X-coordinate of vertex 0 of each simplices.
  ! y0 : array
  !   Y-coordinate of vertex 0 of each simplices.
  ! f0 : array
  !   Field values at vertex 0 of each simplices.
  ! delf0 : array
  !   The 2D difference in each simplices.
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
  real(kind=dp), intent(in) :: x(npart), y(npart)
  real(kind=dp), intent(in) :: x0(nsimp0), y0(nsimp0), f0(nsimp0), delf0(2*nsimp0)
  real(kind=dp), intent(out) :: f_est(npart)

  integer :: i, j

  do i = 1, npart
    j = simplices(i) + 1
    f_est(i) = f0(j) + delf0(2*j-1)*(x(i) - x0(j)) + delf0(2*j)*(y(i) - y0(j))
  end do

end subroutine delaunay_estimate_2d
