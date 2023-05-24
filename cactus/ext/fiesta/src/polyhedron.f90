
subroutine tetrahedron_volume(xa, ya, za, xb, yb, zb, xc, yc, zc, xd, yd, zd, volume)

  ! Computes the volume of a tetrahedron from its vertices.
  !
  ! Parameters
  ! ----------
  ! xa : float
  !   X-coordinate of point A.
  ! ya : float
  !   Y-coordinate of point A.
  ! za : float
  !   Z-coordinate of point A.
  ! xb : float
  !   X-coordinate of point B.
  ! yb : float
  !   Y-coordinate of point B.
  ! zb : float
  !   Z-coordinate of point B.
  ! xc : float
  !   X-coordinate of point C.
  ! yc : float
  !   Y-coordinate of point C.
  ! zc : float
  !   Z-coordinate of point C.
  !
  ! Returns
  ! -------
  ! volume : float
  !   Volume of tetrahedron.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! define variables

  real(kind=dp), intent(in) :: xa, ya, za, xb, yb, zb, xc, yc, zc, xd, yd, zd
  real(kind=dp), intent(out) :: volume

  real(kind=dp) :: a, b, c, d, e, f, g, h, i, det

  ! calculates the volume using the determinant method.

  a = xa - xd
  b = ya - yd
  c = za - zd

  d = xb - xd
  e = yb - yd
  f = zb - zd

  g = xc - xd
  h = yc - yd
  i = zc - zd

  det = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h

  volume = abs(det) / 6.

end subroutine tetrahedron_volume
