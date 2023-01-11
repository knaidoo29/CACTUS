! include "pixel_1dto3d.f90"
!
! module union_finder
!   implicit none
!   integer :: isize = 0
!   integer, allocatable, dimension(:) :: clashid1, clashid2
! contains
!   subroutine hoshen_kopelman_3d(binmap, xgridlen, ygridlen, zgridlen, xlen, groupid)
!     !integer, parameter :: dp = kind(1.d0)
!
!     integer, intent(in) :: xlen, xgridlen, ygridlen, zgridlen
!     integer, intent(in) :: binmap(xlen)
!     integer, intent(out) :: groupid(xlen)
!     !integer, intent(out) :: ngroup
!
!     integer :: i, ngroup, whichgroup, xpixid, ypixid, zpixid
!     integer :: xpixid_plus, ypixid_plus, zpixid_plus
!     integer :: pixid_plus_x, pixid_plus_y, pixid_plus_z
!
!     ngroup = -1
!
!     do i=1, xlen
!       groupid(i) = - 1
!     end do
!
!     do i=1, xlen
!       if (binmap(i) .eq. 1) then
!         if (groupid(i) .eq. -1) then
!           ngroup = ngroup + 1
!           groupid(i) = ngroup
!         end if
!         whichgroup = groupid(i)
!         call pix_id_3dto1d_scalar(i-1, ygridlen, zgridlen, xpixid, ypixid, zpixid)
!         xpixid_plus = xpixid + 1
!         if (xpixid_plus == xgridlen) then
!           xpixid_plus = 0
!         end if
!         ypixid_plus = ypixid + 1
!         if (ypixid_plus == ygridlen) then
!           ypixid_plus = 0
!         end if
!         zpixid_plus = zpixid + 1
!         if (zpixid_plus == zgridlen) then
!           zpixid_plus = 0
!         end if
!         call pix_id_1dto3d_scalar(xpixid_plus, ypixid, zpixid, ygridlen, zgridlen, pixid_plus_x)
!         call pix_id_1dto3d_scalar(xpixid, ypixid_plus, zpixid, ygridlen, zgridlen, pixid_plus_y)
!         call pix_id_1dto3d_scalar(xpixid, ypixid, zpixid_plus, ygridlen, zgridlen, pixid_plus_z)
!         pixid_plus_x = pixid_plus_x + 1
!         pixid_plus_y = pixid_plus_y + 1
!         pixid_plus_z = pixid_plus_z + 1
!         if (binmap(pixid_plus_x) .EQ. 1) then
!           if (groupid(pixid_plus_x) == -1) then
!             groupid(pixid_plus_x) = whichgroup
!           else if (whichgroup .NE. groupid(pixid_plus_x)) then
!             !isize = size(clashid1)
!             allocate(clashid1(isize+1))
!             allocate(clashid2(isize+1))
!             if (whichgroup .lt. groupid(pixid_plus_x)) then
!               clashid1(isize+1) = whichgroup
!               clashid2(isize+1) = groupid(pixid_plus_x)
!             else
!               clashid1(isize+1) = groupid(pixid_plus_x)
!               clashid2(isize+1) = whichgroup
!             end if
!             isize = isize + 1
!           end if
!         end if
!         if (binmap(pixid_plus_y) .EQ. 1) then
!           if (groupid(pixid_plus_y) == -1) then
!             groupid(pixid_plus_y) = whichgroup
!           else if (whichgroup .NE. groupid(pixid_plus_y)) then
!             !isize = size(clashid1)
!             allocate(clashid1(isize+1))
!             allocate(clashid2(isize+1))
!             if (whichgroup .lt. groupid(pixid_plus_y)) then
!               clashid1(isize+1) = whichgroup
!               clashid2(isize+1) = groupid(pixid_plus_y)
!             else
!               clashid1(isize+1) = groupid(pixid_plus_y)
!               clashid2(isize+1) = whichgroup
!             end if
!             isize = isize + 1
!           end if
!         end if
!         if (binmap(pixid_plus_z) .EQ. 1) then
!           if (groupid(pixid_plus_z) == -1) then
!             groupid(pixid_plus_z) = whichgroup
!           else if (whichgroup .NE. groupid(pixid_plus_z)) then
!             !isize = size(clashid1)
!             allocate(clashid1(isize+1))
!             allocate(clashid2(isize+1))
!             if (whichgroup .lt. groupid(pixid_plus_z)) then
!               clashid1(isize+1) = whichgroup
!               clashid2(isize+1) = groupid(pixid_plus_z)
!             else
!               clashid1(isize+1) = groupid(pixid_plus_z)
!               clashid2(isize+1) = whichgroup
!             end if
!             isize = isize + 1
!           end if
!         end if
!       end if
!       ngroup = ngroup + 1
!     end do
!   end subroutine hoshen_kopelman_3d
! end module union_finder
