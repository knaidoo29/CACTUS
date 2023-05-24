
subroutine bin_by_index(pix_id, id_weights, id_len, pix_len, pix_val)

  ! Bin weights by pixel index.
  !
  ! Parameters
  ! ----------
  ! pix_id : array(int)
  !   Pixel index.
  ! id_weights : array(float)
  !   Weights of the binning index.
  ! id_len : int
  !   Length of the pixel and weight arrays.
  ! pix_len : int
  !   Length of the pixel grid.
  !
  ! Returns
  ! -------
  ! pix_val : array(float)
  !   Pixel values after binning.

  implicit none

  ! Parameter declarations

  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: id_len, pix_len
  integer, intent(in) :: pix_id(id_len)
  real(kind=dp), intent(in) :: id_weights(id_len)
  real(kind=dp), intent(out) :: pix_val(pix_len)

  integer :: i

  ! Main

  do i = 1, pix_len
    pix_val(i) = 0.
  end do

  do i = 1, id_len
    if ((pix_id(i) .NE. -1) .AND. (pix_id(i) .GE. 0) .AND. (pix_id(i) .LT. pix_len)) then
      pix_val(pix_id(i)+1) = pix_val(pix_id(i)+1) + id_weights(i)
    end if
  end do

end subroutine bin_by_index
