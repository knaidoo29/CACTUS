
subroutine inv2by2(m, invm)

  ! Invert 2 by 2 matrix.
  !
  ! Parameters
  ! ----------
  ! m : array
  !     2 by 2 matrix.
  !
  ! Returns
  ! -------
  ! invm : array
  !     2 by 2 inverse matrix.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! Declare variables.

  real(kind=dp), intent(in) :: m(4)
  real(kind=dp), intent(out) :: invm(4)

  real(kind=dp) :: a, b, c, d, detm

  a = m(1)
  b = m(2)
  c = m(3)
  d = m(4)

  detm = a*d - b*c

  invm(1) = d/detm
  invm(2) = -b/detm
  invm(3) = -c/detm
  invm(4) = a/detm

end subroutine inv2by2


subroutine inv3by3(m, invm)

  ! Invert 3 by 3 matrix.
  !
  ! Parameters
  ! ----------
  ! m : array
  !     3 by 3 matrix.
  !
  ! Returns
  ! -------
  ! invm : array
  !     3 by 3 inverse matrix.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! Declare variables.

  real(kind=dp), intent(in) :: m(9)
  real(kind=dp), intent(out) :: invm(9)

  real(kind=dp) :: a, b, c, d, e, f, g, h, i, detm
  real(kind=dp) :: aa, bb, cc, dd, ee, ff, gg, hh, ii

  a = m(1)
  b = m(2)
  c = m(3)
  d = m(4)
  e = m(5)
  f = m(6)
  g = m(7)
  h = m(8)
  i = m(9)

  aa = e*i - f*h
  bb = -(d*i - f*g)
  cc = d*h - e*g
  dd = -(b*i - c*h)
  ee = a*i - c*g
  ff = -(a*h - b*g)
  gg = b*f - c*e
  hh = -(a*f - c*d)
  ii = a*e - b*d

  detM = a*aa + b*bb + c*cc

  invm(1) = aa / detm
  invm(2) = dd / detm
  invm(3) = gg / detm
  invm(4) = bb / detm
  invm(5) = ee / detm
  invm(6) = hh / detm
  invm(7) = cc / detm
  invm(8) = ff / detm
  invm(9) = ii / detm

end subroutine inv3by3


subroutine eig2by2(m, eig)

  ! Invert 2 by 2 matrix.
  !
  ! Parameters
  ! ----------
  ! m : array
  !     2 by 2 matrix.
  !
  ! Returns
  ! -------
  ! eig : array
  !     Eigenvalues

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! Declare variables.

  real(kind=dp), intent(in) :: m(4)
  real(kind=dp), intent(out) :: eig(2)

  real(kind=dp) :: m00, m01, m10, m11, eig1, eig2

  m00 = m(1)
  m01 = m(2)
  m10 = m(3)
  m11 = m(4)

  eig1 = 0.5*(m00 + m11 + sqrt(m00**2. + m11**2. - 2.*m00*m11 + 4.*m01*m10))
  eig2 = 0.5*(m00 + m11 - sqrt(m00**2. + m11**2. - 2.*m00*m11 + 4.*m01*m10))

  if (eig1 .le. eig2) then
    eig(1) = eig1
    eig(2) = eig2
  else
    eig(1) = eig2
    eig(2) = eig1
  end if

end subroutine eig2by2

subroutine symeig3by3(m, eig)

  ! Invert 3 by 3 symmetric matrix. Make sure this is symmetric otherwise
  ! this method will not work. Following Eigenvalues and eigenvectors for order 3
  ! symmetric matrices: An analytic approach by Siddique, A.B. and Khraishi, T.A..
  !
  ! Parameters
  ! ----------
  ! m : array
  !     3 by 3 matrix.
  !
  ! Returns
  ! -------
  ! eig : array
  !     Eigenvalues

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! Declare variables.

  real(kind=dp), intent(in) :: m(9)
  real(kind=dp), intent(out) :: eig(3)

  real(kind=dp) :: m00, m01, m02, m11, m12, m22, eig1, eig2, eig3, eigtemp
  real(kind=dp) :: pi, alpha, beta, gamma, p, q, phi

  pi = 4*atan(1.d0)

  m00 = m(1)
  m01 = m(2)
  m02 = m(3)
  m11 = m(5)
  m12 = m(6)
  m22 = m(9)

  alpha = m00 + m11 + m22
  beta = m01**2. + m02**2. + m12**2. - m00*m11 - m11*m22 - m22*m00
  gamma = m00*m11*m22 + 2.*m01*m12*m02 - m00*m12**2. - m22*m01**2. - m11*m02**2.

  p = - (3.*beta + alpha**2.)/3.
  q = - (gamma + (2./27.)*alpha**3. + alpha*beta/3.)
  phi = acos(-q/(2.*((abs(p)/3.)**(1.5))))

  eig1 = alpha/3. + 2.*sqrt(abs(p)/3.)*cos(phi/3.)
  eig2 = alpha/3. - 2.*sqrt(abs(p)/3.)*cos((phi - pi)/3.)
  eig3 = alpha/3. - 2.*sqrt(abs(p)/3.)*cos((phi + pi)/3.)

  if (eig2 .lt. eig1) then
    eigtemp = eig1
    eig1 = eig2
    eig2 = eigtemp
  end if

  if (eig3 .lt. eig2) then
    eigtemp = eig2
    eig2 = eig3
    eig3 = eigtemp
  end if

  if (eig2 .lt. eig1) then
    eigtemp = eig1
    eig1 = eig2
    eig2 = eigtemp
  end if

  eig(1) = eig1
  eig(2) = eig2
  eig(3) = eig3

end subroutine symeig3by3
