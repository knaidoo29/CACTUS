
subroutine pix_id_1dto2d_scalar(xpix_id, ypix_id, ygrid, pix_id)

  ! Returns the combined 2d grid pixel from pixel indices across each axis.
  !
  ! Parameters
  ! ----------
  ! xpix_id : int
  !   Pixel indices along the x axis grid.
  ! ypix_id : int
  !   Pixel indices along the y axis grid.
  ! ygrid : int
  !   Length of y axis grid.
  !
  ! Returns
  ! -------
  ! pix_id : int
  !   2d pixel index.

  implicit none

  integer, intent(in) :: ygrid
  integer, intent(in) :: xpix_id, ypix_id
  integer, intent(out) :: pix_id

  ! Main

  if ((xpix_id .NE. -1) .AND. (ypix_id .NE. -1)) then
    pix_id = ypix_id + ygrid*xpix_id
  else
    pix_id = -1
  end if

end subroutine pix_id_1dto2d_scalar


subroutine pix_id_2dto1d_scalar(pix_id, ygrid, xpix_id, ypix_id)

  ! Returns the 1d grid pixel from 2d pixel indices.
  !
  ! Parameters
  ! ----------
  ! pix_id : int
  !   2d pixel index.
  ! ygrid : int
  !   Length of y axis grid.
  !
  ! Returns
  ! -------
  ! xpix_id : int
  !   Pixel indices along the x axis grid.
  ! ypix_id : int
  !   Pixel indices along the y axis grid.

  implicit none

  integer, intent(in) :: ygrid, pix_id
  integer, intent(out) :: xpix_id, ypix_id

  ! Main

  if (pix_id .EQ. -1) then
    xpix_id = -1
    ypix_id = -1
  else
    xpix_id = int(floor(real(pix_id/ygrid)))
    ypix_id = pix_id - ygrid*xpix_id
  end if

end subroutine pix_id_2dto1d_scalar


subroutine pix_id_1dto2d_grid(xpix_id, ypix_id, xlen, ylen, ygrid, pix_id)

  ! Maps pixels given along a single axis in x and y onto a 2d grid flattened.
  !
  ! Parameters
  ! ----------
  ! xpix_id : int
  !   Pixel indices along the x axis grid.
  ! ypix_id : int
  !   Pixel indices along the y axis grid.
  ! xlen : int
  !   Size of the xpix_id.
  ! ylen : int
  !   Size of the ypix_id.
  ! ygrid : int
  !   Length of y axis grid.
  !
  ! Returns
  ! -------
  ! pix_id : int
  !   2d grid pixel.

  implicit none

  integer, intent(in) :: xlen, ylen, ygrid
  integer, intent(in) :: xpix_id(xlen), ypix_id(ylen)
  integer, intent(out) :: pix_id(xlen*ylen)

  integer :: i, j, ii

  ! Main

  ii = 1

  do i = 1, xlen
    do j = 1, ylen
      call pix_id_1dto2d_scalar(xpix_id(i), ypix_id(j), ygrid, pix_id(ii))
      ii = ii + 1
    end do
  end do

end subroutine pix_id_1dto2d_grid


subroutine pix_id_1dto2d_array(xpix_id, ypix_id, xlen, ygrid, pix_id)

  ! Returns the combined grid pixel from pixel indices across each axis.
  !
  ! Parameters
  ! ----------
  ! xpix_id : int
  !   Pixel indices along the x axis grid.
  ! ypix_id : int
  !   Pixel indices along the y axis grid.
  ! xlen : int
  !   Size of the xpix_id and ypix_id.
  ! ygrid : int
  !   Length of y axis grid.
  !
  ! Returns
  ! -------
  ! pix_id : int
  !   2d array pixel.

  implicit none

  integer, intent(in) :: xlen, ygrid
  integer, intent(in) :: xpix_id(xlen), ypix_id(xlen)
  integer, intent(out) :: pix_id(xlen)

  integer :: i

  ! Main

  do i = 1, xlen
    call pix_id_1dto2d_scalar(xpix_id(i), ypix_id(i), ygrid, pix_id(i))
  end do

end subroutine pix_id_1dto2d_array


subroutine pix_id_2dto1d_array(pix_id, xlen, ygrid, xpix_id, ypix_id)

  ! Returns the combined grid pixel from pixel indices across each axis.
  !
  ! Parameters
  ! ----------
  ! pix_id : int
  !   2d array pixel.
  ! xlen : int
  !   Size of the xpix_id and ypix_id.
  ! ygrid : int
  !   Length of y axis grid.
  !
  ! Returns
  ! -------
  ! xpix_id : int
  !   Pixel indices along the x axis grid.
  ! ypix_id : int
  !   Pixel indices along the y axis grid.

  implicit none

  integer, intent(in) :: xlen, ygrid
  integer, intent(in) :: pix_id(xlen)
  integer, intent(out) :: xpix_id(xlen), ypix_id(xlen)

  integer :: i

  ! Main

  do i = 1, xlen
    call pix_id_2dto1d_scalar(pix_id(i), ygrid, xpix_id(i), ypix_id(i))
  end do

end subroutine pix_id_2dto1d_array
