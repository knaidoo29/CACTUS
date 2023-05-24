
subroutine pix_id_1dto3d_scalar(xpix_id, ypix_id, zpix_id, ygrid, zgrid, pix_id)

  ! Returns the combined 3d grid pixel from pixel indices across each axis.
  !
  ! Parameters
  ! ----------
  ! xpix_id : int
  !   Pixel indices along the x axis grid.
  ! ypix_id : int
  !   Pixel indices along the y axis grid.
  ! zpix_id : int
  !   Pixel indices along the z axis grid.
  ! ygrid : int
  !   Length of y axis grid.
  ! zgrid : int
  !   Length of z axis grid.
  !
  ! Returns
  ! -------
  ! pix_id : int
  !   3d pixel index.

  implicit none

  integer, intent(in) :: ygrid, zgrid
  integer, intent(in) :: xpix_id, ypix_id, zpix_id
  integer, intent(out) :: pix_id

  ! Main

  if ((xpix_id .NE. -1) .AND. (ypix_id .NE. -1) .AND. (zpix_id .NE. -1)) then
    pix_id = zpix_id + zgrid*(ypix_id + ygrid*xpix_id)
  else
    pix_id = -1
  end if

end subroutine pix_id_1dto3d_scalar


subroutine pix_id_3dto1d_scalar(pix_id, ygrid, zgrid, xpix_id, ypix_id, zpix_id)

  ! Returns the combined 3d grid pixel from pixel indices across each axis.
  !
  ! Parameters
  ! ----------
  ! pix_id : int
  !   3d pixel index.
  ! ygrid : int
  !   Length of y axis grid.
  ! zgrid : int
  !   Length of z axis grid.
  !
  ! Returns
  ! -------
  ! xpix_id : int
  !   Pixel indices along the x axis grid.
  ! ypix_id : int
  !   Pixel indices along the y axis grid.
  ! zpix_id : int
  !   Pixel indices along the z axis grid.

  implicit none

  integer, intent(in) :: ygrid, zgrid
  integer, intent(in) :: pix_id
  integer, intent(out) :: xpix_id, ypix_id, zpix_id
  integer :: xypix_id

  ! Main

  if (pix_id .EQ. -1) then
    xpix_id = -1
    ypix_id = -1
    zpix_id = -1
    xypix_id = 0
  else
    xypix_id = int(floor(real(pix_id/zgrid)))
    zpix_id = pix_id - zgrid*xypix_id
    xpix_id = int(floor(real(xypix_id/ygrid)))
    ypix_id = xypix_id - ygrid*xpix_id
  end if

end subroutine pix_id_3dto1d_scalar


subroutine pix_id_1dto3d_grid(xpix_id, ypix_id, zpix_id, xlen, ylen, zlen &
  , ygrid, zgrid, pix_id)

  ! Maps pixels given along a single axis in x, y and z onto a 3d grid
  ! flattened.
  !
  ! Parameters
  ! ----------
  ! xpix_id : int
  !   Pixel indices along the x axis grid.
  ! ypix_id : int
  !   Pixel indices along the y axis grid.
  ! zpix_id : int
  !   Pixel indices along the z axis grid.
  ! xlen : int
  !   Size of the xpix_id.
  ! ylen : int
  !   Size of the ypix_id.
  ! zlen : int
  !   Size of the zpix_id.
  ! ygrid : int
  !   Length of y axis grid.
  ! zgrid : int
  !   Length of z axis grid.
  !
  ! Returns
  ! -------
  ! pix_id : int
  !   3d grid pixel.

  implicit none

  integer, intent(in) :: xlen, ylen, zlen, ygrid, zgrid
  integer, intent(in) :: xpix_id(xlen), ypix_id(ylen), zpix_id(zlen)
  integer, intent(out) :: pix_id(xlen*ylen*zlen)

  integer :: i, j, k, ii

  ! Main

  ii = 1

  do i = 1, xlen
    do j = 1, ylen
      do k = 1, zlen
        call pix_id_1dto3d_scalar(xpix_id(i), ypix_id(j), zpix_id(k), ygrid, zgrid, pix_id(ii))
        ii = ii + 1
      end do
    end do
  end do

end subroutine pix_id_1dto3d_grid


subroutine pix_id_1dto3d_array(xpix_id, ypix_id, zpix_id, xlen, ygrid, zgrid, pix_id)

  ! Returns the combined grid pixel from pixel indices across each axis.
  !
  ! Parameters
  ! ----------
  ! xpix_id : int
  !   Pixel indices along the x axis grid.
  ! ypix_id : int
  !   Pixel indices along the y axis grid.
  ! zpix_id : int
  !   Pixel indices along the z axis grid.
  ! xlen : int
  !   Size of the xpix_id, ypix_id and zpix_id.
  ! ygrid : int
  !   Length of y axis grid.
  ! zgrid : int
  !   Length of z axis grid.
  !
  ! Returns
  ! -------
  ! pix_id : int
  !   3d array pixel.

  implicit none

  integer, intent(in) :: xlen, ygrid, zgrid
  integer, intent(in) :: xpix_id(xlen), ypix_id(xlen), zpix_id(xlen)
  integer, intent(out) :: pix_id(xlen)

  integer :: i

  ! Main

  do i = 1, xlen
    call pix_id_1dto3d_scalar(xpix_id(i), ypix_id(i), zpix_id(i), ygrid, zgrid, pix_id(i))
  end do

end subroutine pix_id_1dto3d_array



subroutine pix_id_3dto1d_array(pix_id, xlen, ygrid, zgrid, xpix_id, ypix_id, zpix_id)

  ! Returns pixel indices across each axis from the combined grid pixel.
  !
  ! Parameters
  ! ----------
  ! pix_id : int
  !   3d array pixel.
  ! xlen : int
  !   Size of the xpix_id, ypix_id and zpix_id.
  ! ygrid : int
  !   Length of y axis grid.
  ! zgrid : int
  !   Length of z axis grid.
  !
  ! Returns
  ! -------
  ! xpix_id : int
  !   Pixel indices along the x axis grid.
  ! ypix_id : int
  !   Pixel indices along the y axis grid.
  ! zpix_id : int
  !   Pixel indices along the z axis grid.

  implicit none

  integer, intent(in) :: xlen, ygrid, zgrid
  integer, intent(in) :: pix_id(xlen)
  integer, intent(out) :: xpix_id(xlen), ypix_id(xlen), zpix_id(xlen)

  integer :: i

  ! Main

  do i = 1, xlen
    call pix_id_3dto1d_scalar(pix_id(i), ygrid, zgrid, xpix_id(i), ypix_id(i), zpix_id(i))
  end do

end subroutine pix_id_3dto1d_array
