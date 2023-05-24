
subroutine dfdx_1d_periodic(f, boxsize, ngrid, df)

  ! Numerical differentiation assuming a uniform grid with periodic boundaries in 1D.
  !
  ! Parameters
  ! ----------
  ! f : array
  !   Field values on the grid.
  ! boxsize : float
  !   Size of the grid.
  ! ngrid : int
  !   Number of points on the grid.
  !
  ! Returns
  ! -------
  ! df : array
  !   Differentiated function along x.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: ngrid
  real(kind=dp), intent(in) :: f(ngrid), boxsize
  real(kind=dp), intent(out) :: df(ngrid)

  real(kind=dp) :: dl

  integer :: i, i1, i2

  dl = boxsize / real(ngrid)

  do i = 0, ngrid - 1

    if (i .eq. 0) then
      i1 = ngrid - 1
    else
      i1 = i - 1
    end if

    if (i .eq. ngrid - 1) then
      i2 = 0
    else
      i2 = i + 1
    end if

    df(i + 1) = (f(i2 + 1) - f(i1 + 1))/(2*dl)

  end do

end subroutine dfdx_1d_periodic


subroutine dfdx_2d_periodic(f, boxsize, ngrid, df)

  ! Numerical differentiation in x assuming a uniform grid with periodic boundaries in 2D.
  !
  ! Parameters
  ! ----------
  ! f : array
  !   Field values on the grid.
  ! boxsize : float
  !   Size of the grid.
  ! ngrid : int
  !   Number of points on the grid.
  !
  ! Returns
  ! -------
  ! df : array
  !   Differentiated function along x.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: ngrid
  real(kind=dp), intent(in) :: f(ngrid*ngrid), boxsize
  real(kind=dp), intent(out) :: df(ngrid*ngrid)

  real(kind=dp) :: dl

  integer :: i, i1, i2, j

  dl = boxsize / real(ngrid)

  do i = 0, ngrid - 1

    if (i .eq. 0) then
      i1 = ngrid - 1
    else
      i1 = i - 1
    end if

    if (i .eq. ngrid - 1) then
      i2 = 0
    else
      i2 = i + 1
    end if

    do j = 0, ngrid - 1
      df(j + ngrid*i + 1) = (f(j + ngrid*i2 + 1) - f(j + ngrid*i1 + 1))/(2*dl)
    end do

  end do

end subroutine dfdx_2d_periodic


subroutine dfdy_2d_periodic(f, boxsize, ngrid, df)

  ! Numerical differentiation in y assuming a uniform grid with periodic boundaries in 2D.
  !
  ! Parameters
  ! ----------
  ! f : array
  !   Field values on the grid.
  ! boxsize : float
  !   Size of the grid.
  ! ngrid : int
  !   Number of points on the grid.
  !
  ! Returns
  ! -------
  ! df : array
  !   Differentiated function along x.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: ngrid
  real(kind=dp), intent(in) :: f(ngrid*ngrid), boxsize
  real(kind=dp), intent(out) :: df(ngrid*ngrid)

  real(kind=dp) :: dl

  integer :: i, j1, j2, j

  dl = boxsize / real(ngrid)

  do j = 0, ngrid - 1

    if (j .eq. 0) then
      j1 = ngrid - 1
    else
      j1 = j - 1
    end if

    if (j .eq. ngrid - 1) then
      j2 = 0
    else
      j2 = j + 1
    end if

    do i = 0, ngrid - 1
      df(j + ngrid*i + 1) = (f(j2 + ngrid*i + 1) - f(j1 + ngrid*i + 1))/(2*dl)
    end do

  end do

end subroutine dfdy_2d_periodic


subroutine dfdx_3d_periodic(f, boxsize, ngrid, df)

  ! Numerical differentiation in x assuming a uniform grid with periodic boundaries in 3D.
  !
  ! Parameters
  ! ----------
  ! f : array
  !   Field values on the grid.
  ! boxsize : float
  !   Size of the grid.
  ! ngrid : int
  !   Number of points on the grid.
  !
  ! Returns
  ! -------
  ! df : array
  !   Differentiated function along x.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: ngrid
  real(kind=dp), intent(in) :: f(ngrid*ngrid*ngrid), boxsize
  real(kind=dp), intent(out) :: df(ngrid*ngrid*ngrid)

  real(kind=dp) :: dl

  integer :: i, i1, i2, j, k

  dl = boxsize / real(ngrid)

  do i = 0, ngrid - 1

    if (i .eq. 0) then
      i1 = ngrid - 1
    else
      i1 = i - 1
    end if

    if (i .eq. ngrid - 1) then
      i2 = 0
    else
      i2 = i + 1
    end if

    do j = 0, ngrid - 1
      do k = 0, ngrid - 1
        df(k + ngrid*(j + ngrid*i) + 1) = (f(k + ngrid*(j + ngrid*i2) + 1) - f(k + ngrid*(j + ngrid*i1) + 1))/(2*dl)
      end do
    end do

  end do

end subroutine dfdx_3d_periodic


subroutine dfdy_3d_periodic(f, boxsize, ngrid, df)

  ! Numerical differentiation in x assuming a uniform grid with periodic boundaries in 3D.
  !
  ! Parameters
  ! ----------
  ! f : array
  !   Field values on the grid.
  ! boxsize : float
  !   Size of the grid.
  ! ngrid : int
  !   Number of points on the grid.
  !
  ! Returns
  ! -------
  ! df : array
  !   Differentiated function along x.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: ngrid
  real(kind=dp), intent(in) :: f(ngrid*ngrid*ngrid), boxsize
  real(kind=dp), intent(out) :: df(ngrid*ngrid*ngrid)

  real(kind=dp) :: dl

  integer :: i, j1, j2, j, k

  dl = boxsize / real(ngrid)

  do j = 0, ngrid - 1

    if (j .eq. 0) then
      j1 = ngrid - 1
    else
      j1 = j - 1
    end if

    if (j .eq. ngrid - 1) then
      j2 = 0
    else
      j2 = j + 1
    end if

    do i = 0, ngrid - 1
      do k = 0, ngrid - 1
        df(k + ngrid*(j + ngrid*i) + 1) = (f(k + ngrid*(j2 + ngrid*i) + 1) - f(k + ngrid*(j1 + ngrid*i) + 1))/(2*dl)
      end do
    end do

  end do

end subroutine dfdy_3d_periodic


subroutine dfdz_3d_periodic(f, boxsize, ngrid, df)

  ! Numerical differentiation in x assuming a uniform grid with periodic boundaries in 3D.
  !
  ! Parameters
  ! ----------
  ! f : array
  !   Field values on the grid.
  ! boxsize : float
  !   Size of the grid.
  ! ngrid : int
  !   Number of points on the grid.
  !
  ! Returns
  ! -------
  ! df : array
  !   Differentiated function along x.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: ngrid
  real(kind=dp), intent(in) :: f(ngrid*ngrid*ngrid), boxsize
  real(kind=dp), intent(out) :: df(ngrid*ngrid*ngrid)

  real(kind=dp) :: dl

  integer :: i, k1, k2, j, k

  dl = boxsize / real(ngrid)

  do k = 0, ngrid - 1

    if (k .eq. 0) then
      k1 = ngrid - 1
    else
      k1 = k - 1
    end if

    if (k .eq. ngrid - 1) then
      k2 = 0
    else
      k2 = k + 1
    end if

    do j = 0, ngrid - 1
      do i = 0, ngrid - 1
        df(k + ngrid*(j + ngrid*i) + 1) = (f(k2 + ngrid*(j + ngrid*i) + 1) - f(k1 + ngrid*(j + ngrid*i) + 1))/(2*dl)
      end do
    end do

  end do

end subroutine dfdz_3d_periodic
