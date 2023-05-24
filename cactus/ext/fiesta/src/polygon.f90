
subroutine triangle_area(xa, ya, xb, yb, xc, yc, area)

  ! Determines the area of a triangle.
  !
  ! Parameters
  ! ----------
  ! xa : float
  !   X coordinate of point a.
  ! ya : float
  !   Y coordinate of point a.
  ! xb : float
  !   X coordinate of point b.
  ! yb : float
  !   Y coordinate of point b.
  ! xc : float
  !   X coordinate of point c.
  ! yc : float
  !   Y coordinate of point c.
  !
  ! Returns
  ! -------
  ! area : float
  !   Area of the triangle.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! define variables

  real(kind=dp), intent(in) :: xa, ya, xb, yb, xc, yc
  real(kind=dp), intent(out) :: area

  ! Calculate area.

  area = 0.5*abs(xa*(yb-yc) + xb*(yc - ya) + xc*(ya - yb))

end subroutine triangle_area


subroutine sum_triangle_area(xas, yas, xbs, ybs, xcs, ycs, ntri, area)

  ! Determines the area of a triangle.
  !
  ! Parameters
  ! ----------
  ! xas : array
  !   X coordinates of points a.
  ! yas : array
  !   Y coordinates of points a.
  ! xbs : array
  !   X coordinates of points b.
  ! ybs : array
  !   Y coordinates of points b.
  ! xcs : array
  !   X coordinates of points c.
  ! ycs : array
  !   Y coordinates of points c.
  ! ntri : int
  !   Number of triangles
  !
  ! Returns
  ! -------
  ! area : float
  !   Total area of the triangles.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! define variables

  integer, intent(in) :: ntri
  real(kind=dp), intent(in) :: xas(ntri), yas(ntri), xbs(ntri), ybs(ntri), xcs(ntri), ycs(ntri)
  real(kind=dp), intent(out) :: area

  integer :: i
  real(kind=dp) :: area_tri

  ! loop over triangles and calculate the areas.

  area = 0.

  do i = 1, ntri
    call triangle_area(xas(i), yas(i), xbs(i), ybs(i), xcs(i), ycs(i), area_tri)
    area = area + area_tri
  end do

end subroutine sum_triangle_area
