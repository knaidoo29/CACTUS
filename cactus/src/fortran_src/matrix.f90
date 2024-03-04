
subroutine eig2by2_single(m00, m01, m10, m11, eig1, eig2)

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

  real(kind=dp), intent(in) :: m00, m01, m10, m11
  real(kind=dp), intent(out) :: eig1, eig2

  real(kind=dp) :: eigtemp

  eig1 = 0.5*(m00 + m11 + sqrt(m00**2. + m11**2. - 2.*m00*m11 + 4.*m01*m10))
  eig2 = 0.5*(m00 + m11 - sqrt(m00**2. + m11**2. - 2.*m00*m11 + 4.*m01*m10))

  if (eig2 .le. eig1) then
    eigtemp = eig1
    eig1 = eig2
    eig2 = eigtemp
  end if

end subroutine eig2by2_single


subroutine eig2by2_array(m00, m01, m10, m11, mlen, eig1, eig2)

  ! Invert 2 by 2 symmetric matrix. Make sure this is symmetric otherwise
  ! this method will not work. Following Eigenvalues and eigenvectors for order 3
  ! symmetric matrices: An analytic approach by Siddique, A.B. and Khraishi, T.A..
  !
  ! Parameters
  ! ----------
  ! m00, m01, m10, m11 : float
  !     Matrix components.
  !
  ! Returns
  ! -------
  ! eig1, eig2 : float
  !     Eigenvalues

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! Declare variables.

  integer, intent(in) :: mlen
  real(kind=dp), intent(in) :: m00(mlen), m01(mlen), m10(mlen), m11(mlen)
  real(kind=dp), intent(out) :: eig1(mlen), eig2(mlen)

  integer :: i

  do i = 1, mlen
    call eig2by2_single(m00(i), m01(i), m10(i), m11(i), eig1(i), eig2(i))
  end do

end subroutine eig2by2_array


subroutine sym_eig3by3_single(m00, m01, m02, m11, m12, m22, eig1, eig2, eig3)

  ! Invert 3 by 3 symmetric matrix. Make sure this is symmetric otherwise
  ! this method will not work. Following Eigenvalues and eigenvectors for order 3
  ! symmetric matrices: An analytic approach by Siddique, A.B. and Khraishi, T.A..
  !
  ! Parameters
  ! ----------
  ! m00, m01, m02, m11, m12, m22 : float
  !     Upper triangle of the symmetric components.
  !
  ! Returns
  ! -------
  ! eig1, eig2, eig3 : float
  !     Eigenvalues

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! Declare variables.

  real(kind=dp), intent(in) :: m00, m01, m02, m11, m12, m22
  real(kind=dp), intent(out) :: eig1, eig2, eig3

  real(kind=dp) :: eigtemp
  real(kind=dp) :: pi, alpha, beta, gamma, p, q, phi

  pi = 4*atan(1.d0)

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

end subroutine sym_eig3by3_single


subroutine sym_eig3by3_array(m00, m01, m02, m11, m12, m22, mlen, eig1, eig2, eig3)

  ! Invert 3 by 3 symmetric matrix. Make sure this is symmetric otherwise
  ! this method will not work. Following Eigenvalues and eigenvectors for order 3
  ! symmetric matrices: An analytic approach by Siddique, A.B. and Khraishi, T.A..
  !
  ! Parameters
  ! ----------
  ! m00, m01, m02, m11, m12, m22 : float
  !     Upper triangle of the symmetric components.
  !
  ! Returns
  ! -------
  ! eig1, eig2, eig3 : float
  !     Eigenvalues

  implicit none
  integer, parameter :: dp = kind(1.d0)

  ! Declare variables.

  integer, intent(in) :: mlen
  real(kind=dp), intent(in) :: m00(mlen), m01(mlen), m02(mlen), m11(mlen), m12(mlen), m22(mlen)
  real(kind=dp), intent(out) :: eig1(mlen), eig2(mlen), eig3(mlen)

  integer :: i

  do i = 1, mlen
    call sym_eig3by3_single(m00(i), m01(i), m02(i), m11(i), m12(i), m22(i), eig1(i), eig2(i), eig3(i))
  end do

end subroutine sym_eig3by3_array
