include "pixel_util.f90"
include "pixel_1dto2d.f90"
include "pixel_1dto3d.f90"


subroutine find_maxint(len, array, maxvalue)

  implicit none

  integer, intent(in) :: len
  integer, intent(in) :: array(len)
  integer, intent(out) :: maxvalue
  integer :: i

  maxvalue = array(1)

  do i = 2, len
    if (array(i) .gt. maxvalue) then
      maxvalue = array(i)
    end if
  end do

end subroutine find_maxint


subroutine cascade(lenlabels, labels, indexin, indexout)

  ! Cascade label index.
  !
  ! Parameters
  ! ----------
  ! lenlabels : int
  !   Length of label array.
  ! labels : int array
  !   Label array.
  ! indexin : int
  !   Label index in.
  !
  ! Returns
  ! -------
  ! indexout : out
  !   Cascaded label index out.

  implicit none

  integer, intent(in) :: lenlabels, labels(lenlabels), indexin
  integer, intent(out) :: indexout

  indexout = indexin

  do while (labels(indexout) .ne. indexout)
    indexout = labels(indexout)
  end do

end subroutine cascade


subroutine cascade_all(maxlabel, lenlabels, labels, labelsout)

  ! Cascade label index for an array.
  !
  ! Parameters
  ! ----------
  ! maxlabel : int
  !   Maximum label.
  ! lenlabels : int
  !   Length of label array.
  ! labels : int array
  !   Label array.
  !
  ! Returns
  ! -------
  ! labelsout : int array
  !   Cascade all label index in an array.

  implicit none

  integer, intent(in) :: maxlabel, lenlabels, labels(lenlabels)
  integer, intent(out) :: labelsout(lenlabels)

  integer :: i, j

  do i = 1, maxlabel
    call cascade(lenlabels, labels, labels(i), j)
    labelsout(i) = j
  end do

end subroutine cascade_all


subroutine unionise(ind1, ind2, lenlabels, labels, ind1out, ind2out, indout)

  ! Finds the union of two label indexes.
  !
  ! Parameters
  ! ----------
  ! ind1, ind2 : int
  !   Index 1 and 2.
  ! lenlabels : int
  !   Length of label array.
  ! labels : int array
  !   Label array.
  !
  ! Returns
  ! -------
  ! ind1out, ind2out : int
  !   Outputted index 1 and 2.
  ! indout : int
  !   Index out.

  implicit none

  integer, intent(in) :: ind1, ind2, lenlabels
  integer, intent(in) :: labels(lenlabels)
  integer, intent(out) :: ind1out, ind2out, indout

  call cascade(lenlabels, labels, ind1, ind1out)
  call cascade(lenlabels, labels, ind2, ind2out)

  if (ind1out .le. ind2out) then
    indout = ind1out
  else
    indout = ind2out
  end if

end subroutine unionise


subroutine get_nlabels(maxlabel, lenlabels, labels, nlabels)

  ! Finds the union of two label indexes.
  !
  ! Parameters
  ! ----------
  ! maxlabel : int
  !   Maximum label.
  ! lenlabels : int
  !   Length of label array.
  ! labels : int array
  !   Label array.
  !
  ! Returns
  ! -------
  ! nlabels : int array
  !   Number of recurrences for each label index.

  implicit none

  integer, intent(in) :: maxlabel, lenlabels
  integer, intent(in) :: labels(lenlabels)
  integer, intent(out) :: nlabels(maxlabel)

  integer :: i

  do i = 1, maxlabel
    nlabels(i) = 0
  end do

  do i = 1, lenlabels
    if ((labels(i) .ge. 1) .and. (labels(i) .le. maxlabel)) then
      nlabels(labels(i)) = nlabels(labels(i)) + 1
    end if
  end do

end subroutine get_nlabels


subroutine shuffle_down(maxlabel, lenlabels, labels, newmaxlabel, labelsout)

  ! Shuffle label index for an array.
  !
  ! Parameters
  ! ----------
  ! maxlabel : int
  !   Maximum label.
  ! lenlabels : int
  !   Length of label array.
  ! labels : int array
  !   Label array.
  !
  ! Returns
  ! -------
  ! newmaxlabel : int
  !   Output maximum label.
  ! labelsout : int array
  !   Cascade all label index in an array.

  implicit none

  integer, intent(in) :: maxlabel, lenlabels
  integer, intent(in) :: labels(lenlabels)
  integer, intent(out) :: newmaxlabel, labelsout(lenlabels)

  integer :: i, nlabels(maxlabel), j, maplabel(maxlabel)

  call get_nlabels(maxlabel, lenlabels, labels, nlabels)

  j = 0
  do i = 1, maxlabel
    if (nlabels(i) .gt. 0) then
      j = j + 1
      maplabel(i) = j
    else
      maplabel(i) = 0
    end if
  end do

  newmaxlabel = j

  do i = 1, maxlabel
    labelsout(i) = maplabel(labels(i))
  end do

end subroutine shuffle_down


subroutine order_label_index(maxlabel, lenlabels, labels, start_index &
  , end_index, nlabels, orderlabels, indexlabels)

  ! Order array index by label index.
  !
  ! Parameters
  ! ----------
  ! maxlabel : int
  !   Maximum label.
  ! lenlabels : int
  !   Length of label array.
  ! labels : int array
  !   Label array.
  !
  ! Returns
  ! -------
  ! start_index, end_index : int array
  !   Start and end for each label index
  ! nlabels : int array
  !   Number of recurrences for each label index.
  ! orderlabels : int array
  !   Ordered label values.
  ! indexlabels : int array
  !   Ordered indexes for each label.

  implicit none

  integer, intent(in) :: maxlabel, lenlabels
  integer, intent(in) :: labels(lenlabels)
  integer, intent(out) :: orderlabels(lenlabels), indexlabels(lenlabels)
  integer, intent(out) :: start_index(maxlabel), end_index(maxlabel), nlabels(maxlabel)
  integer :: i, j, total_labeled

  call get_nlabels(maxlabel, lenlabels, labels, nlabels)

  total_labeled = 0
  do i = 1, maxlabel
    total_labeled = total_labeled + nlabels(i)
  end do

  j = 1
  do i = 1, maxlabel
    if (nlabels(i) .ne. 0) then
      start_index(i) = j
      end_index(i) = j
      j = j + nlabels(i)
    else
      start_index(i) = 0
      end_index(i) = 0
    end if
  end do

  do i = 1, lenlabels
    orderlabels(end_index(labels(i))) = labels(i)
    indexlabels(end_index(labels(i))) = i
    end_index(labels(i)) = end_index(labels(i)) + 1
  end do

end subroutine order_label_index


subroutine hoshen_kopelman_2d(binmap, binlen, nxgrid, nygrid, periodx, periody, maxlabel, group)

  ! Hoshen-Kopelman algorithm in 2D.
  !
  ! Parameters
  ! ----------
  ! binmap : int array
  !   Binary array.
  ! nxgrid, nygrid : int
  !   Length of each axis.
  ! periodx, periody : bool
  !   Periodic boundary conditions..
  !
  ! Returns
  ! -------
  ! maxlabel : int
  !   Maximum label.
  ! group : int array
  !   Group ID for each point.

  implicit none

  integer, intent(in) :: nxgrid, nygrid, binlen
  integer, intent(in) :: binmap(binlen)
  logical, intent(in) :: periodx, periody
  integer, intent(out) :: maxlabel, group(binlen)

  integer :: i, j, ix, iy, jx, jy, xpix_id, ypix_id
  integer :: newmaxlabel, labels(binlen)
  integer :: xpix_plus_id, ypix_plus_id
  integer :: i_px, i_py
  integer :: ind1out, ind2out, indout
  integer :: labelsout1(binlen)

  maxlabel = 0

  do i = 1, binlen
    group(i) = 0
    labels(i) = 0
    labelsout1(i) = 0
  end do

  do i = 1, binlen
    if (binmap(i) .eq. 1) then

      if (group(i) .eq. 0) then
        maxlabel = maxlabel + 1
        labels(maxlabel) = maxlabel
        group(i) = maxlabel
      end if

      call pix_f2p(i, j)
      call pix_id_2dto1d_scalar(j, nygrid, ix, iy)
      call pix_p2f(ix, xpix_id)
      call pix_p2f(iy, ypix_id)

      if (xpix_id .eq. nxgrid) then
        if (periodx .eqv. .true.) then
          xpix_plus_id = 1
        else
          xpix_plus_id = 0
        end if
      else
        xpix_plus_id = xpix_id + 1
      end if

      if (ypix_id .eq. nygrid) then
        if (periody .eqv. .true.) then
          ypix_plus_id = 1
        else
          ypix_plus_id = 0
        end if
      else
        ypix_plus_id = ypix_id + 1
      end if

      if (xpix_plus_id .ne. 0) then
        call pix_f2p(xpix_plus_id, jx)
        call pix_f2p(ypix_id, jy)
        call pix_id_1dto2d_scalar(jx, jy, nygrid, ix)
        call pix_p2f(ix, i_px)
        if (binmap(i_px) .eq. 1) then
          if (group(i_px) .eq. 0) then
            group(i_px) = group(i)
          else if (group(i_px) .ne. group(i)) then
            call unionise(group(i), group(i_px), binlen, labels, ind1out, ind2out, indout)
            labels(ind1out) = indout
            labels(ind2out) = indout
            labels(group(i)) = indout
            labels(group(i_px)) = indout
            group(i) = indout
            group(i_px) = indout
          end if
        end if
      end if

      if (ypix_plus_id .ne. 0) then
        call pix_f2p(xpix_id, jx)
        call pix_f2p(ypix_plus_id, jy)
        call pix_id_1dto2d_scalar(jx, jy, nygrid, iy)
        call pix_p2f(iy, i_py)
        if (binmap(i_py) .eq. 1) then
          if (group(i_py) .eq. 0) then
            group(i_py) = group(i)
          else if (group(i_py) .ne. group(i)) then
            call unionise(group(i), group(i_py), binlen, labels, ind1out, ind2out, indout)
            labels(ind1out) = indout
            labels(ind2out) = indout
            labels(group(i)) = indout
            labels(group(i_py)) = indout
            group(i) = indout
            group(i_py) = indout
          end if
        end if
      end if

    end if
  end do

  call cascade_all(maxlabel, binlen, labels, labelsout1)
  call shuffle_down(maxlabel, binlen, labelsout1, newmaxlabel, labels)

  do i = 1, binlen
    if (group(i) .ne. 0) then
      group(i) = labels(group(i))
    end if
  end do

  maxlabel = newmaxlabel

end subroutine hoshen_kopelman_2d


subroutine hoshen_kopelman_3d(binmap, binlen, nxgrid, nygrid, nzgrid, periodx, periody &
  , periodz, maxlabel, group)

  ! Hoshen-Kopelman algorithm in 3D.
  !
  ! Parameters
  ! ----------
  ! binmap : int array
  !   Binary array.
  ! nxgrid, nygrid, nzgrid : int
  !   Length of each axis.
  ! periodx, periody, periodz : bool
  !   Periodic boundary conditions..
  !
  ! Returns
  ! -------
  ! maxlabel : int
  !   Maximum label.
  ! group : int array
  !   Group ID for each point.

  implicit none

  integer, intent(in) :: nxgrid, nygrid, nzgrid, binlen
  integer, intent(in) :: binmap(binlen)
  logical, intent(in) :: periodx, periody, periodz
  integer, intent(out) :: maxlabel, group(binlen)

  integer :: i, j, ix, iy, iz, jx, jy, jz, xpix_id, ypix_id, zpix_id
  integer :: newmaxlabel, labels(binlen)
  integer :: xpix_plus_id, ypix_plus_id, zpix_plus_id
  integer :: i_px, i_py, i_pz
  integer :: ind1out, ind2out, indout
  integer :: labelsout1(binlen)

  maxlabel = 0

  do i = 1, binlen
    group(i) = 0
    labels(i) = 0
    labelsout1(i) = 0
  end do

  do i = 1, binlen
    if (binmap(i) .eq. 1) then

      if (group(i) .eq. 0) then
        maxlabel = maxlabel + 1
        labels(maxlabel) = maxlabel
        group(i) = maxlabel
      end if

      call pix_f2p(i, j)
      call pix_id_3dto1d_scalar(j, nygrid, nzgrid, ix, iy, iz)
      call pix_p2f(ix, xpix_id)
      call pix_p2f(iy, ypix_id)
      call pix_p2f(iz, zpix_id)

      if (xpix_id .eq. nxgrid) then
        if (periodx .eqv. .true.) then
          xpix_plus_id = 1
        else
          xpix_plus_id = 0
        end if
      else
        xpix_plus_id = xpix_id + 1
      end if

      if (ypix_id .eq. nygrid) then
        if (periody .eqv. .true.) then
          ypix_plus_id = 1
        else
          ypix_plus_id = 0
        end if
      else
        ypix_plus_id = ypix_id + 1
      end if

      if (zpix_id .eq. nzgrid) then
        if (periodz .eqv. .true.) then
          zpix_plus_id = 1
        else
          zpix_plus_id = 0
        end if
      else
        zpix_plus_id = zpix_id + 1
      end if

      if (xpix_plus_id .ne. 0) then
        call pix_f2p(xpix_plus_id, jx)
        call pix_f2p(ypix_id, jy)
        call pix_f2p(zpix_id, jz)
        call pix_id_1dto3d_scalar(jx, jy, jz, nygrid, nzgrid, ix)
        call pix_p2f(ix, i_px)
        if (binmap(i_px) .eq. 1) then
          if (group(i_px) .eq. 0) then
            group(i_px) = group(i)
          else if (group(i_px) .ne. group(i)) then
            call unionise(group(i), group(i_px), binlen, labels, ind1out, ind2out, indout)
            labels(ind1out) = indout
            labels(ind2out) = indout
            labels(group(i)) = indout
            labels(group(i_px)) = indout
            group(i) = indout
            group(i_px) = indout
          end if
        end if
      end if

      if (ypix_plus_id .ne. 0) then
        call pix_f2p(xpix_id, jx)
        call pix_f2p(ypix_plus_id, jy)
        call pix_f2p(zpix_id, jz)
        call pix_id_1dto3d_scalar(jx, jy, jz, nygrid, nzgrid, iy)
        call pix_p2f(iy, i_py)
        if (binmap(i_py) .eq. 1) then
          if (group(i_py) .eq. 0) then
            group(i_py) = group(i)
          else if (group(i_py) .ne. group(i)) then
            call unionise(group(i), group(i_py), binlen, labels, ind1out, ind2out, indout)
            labels(ind1out) = indout
            labels(ind2out) = indout
            labels(group(i)) = indout
            labels(group(i_py)) = indout
            group(i) = indout
            group(i_py) = indout
          end if
        end if
      end if

      if (zpix_plus_id .ne. 0) then
        call pix_f2p(xpix_id, jx)
        call pix_f2p(ypix_id, jy)
        call pix_f2p(zpix_plus_id, jz)
        call pix_id_1dto3d_scalar(jx, jy, jz, nygrid, nzgrid, iz)
        call pix_p2f(iz, i_pz)
        if (binmap(i_pz) .eq. 1) then
          if (group(i_pz) .eq. 0) then
            group(i_pz) = group(i)
          else if (group(i_pz) .ne. group(i)) then
            call unionise(group(i), group(i_pz), binlen, labels, ind1out, ind2out, indout)
            labels(ind1out) = indout
            labels(ind2out) = indout
            labels(group(i)) = indout
            labels(group(i_pz)) = indout
            group(i) = indout
            group(i_pz) = indout
          end if
        end if
      end if
    end if
  end do

  call cascade_all(maxlabel, binlen, labels, labelsout1)
  call shuffle_down(maxlabel, binlen, labelsout1, newmaxlabel, labels)

  do i = 1, binlen
    if (group(i) .ne. 0) then
      group(i) = labels(group(i))
    end if
  end do

  maxlabel = newmaxlabel

end subroutine hoshen_kopelman_3d


subroutine resolve_clashes(group1, group2, lengroup, labels, lenlabels, labelsout)

  ! Resolve group ID clashes.
  !
  ! Parameters
  ! ----------
  ! group1, group2 : int array
  !   Group ID for the same points.
  ! lengroup : int
  !   Length of the group ID.
  ! labels : int array
  !   Label array.
  ! lenlabels : int
  !   Length of label array.
  !
  ! Returns
  ! -------
  ! labelsout : int array
  !   Label output.

  implicit none

  integer, intent(in) :: lengroup, lenlabels
  integer, intent(in) :: group1(lengroup), group2(lengroup), labels(lenlabels)
  integer, intent(out) :: labelsout(lenlabels)

  integer :: i, indout1, indout2, indout, labelsout1(lenlabels)

  do i = 1, lenlabels
    labelsout1(i) = labels(i)
  end do

  do i = 1, lengroup
    if ((group1(i) .ne. 0) .and. (group2(i) .ne. 0)) then
      call unionise(group1(i), group2(i), lenlabels, labelsout1, indout1, indout2, indout)
      labelsout1(indout1) = indout
      labelsout1(indout2) = indout
      labelsout1(group1(i)) = indout
      labelsout1(group2(i)) = indout
    end if
  end do

  call cascade_all(lenlabels, lenlabels, labelsout1, labelsout)

end subroutine resolve_clashes


subroutine resolve_labels(labels1, labels2, lenlabels, labelsout)

  ! Resolve label ID clashes.
  !
  ! Parameters
  ! ----------
  ! labels1, labels2 : int array
  !   Label array.
  ! lenlabels : int
  !   Length of label array.
  !
  ! Returns
  ! -------
  ! labelsout : int array
  !   Label output.

  implicit none

  integer, intent(in) :: lenlabels
  integer, intent(in) :: labels1(lenlabels), labels2(lenlabels)
  integer, intent(out) :: labelsout(lenlabels)

  integer :: i, indout1, indout2, indout, labelsout1(lenlabels)

  do i = 1, lenlabels
    labelsout1(i) = labels1(i)
  end do

  do i = 1, lenlabels
    call unionise(labels1(i), labels2(i), lenlabels, labelsout1, indout1, indout2, indout)
    labelsout1(indout1) = indout
    labelsout1(indout2) = indout
    labelsout1(labels1(i)) = indout
    labelsout1(labels2(i)) = indout
  end do

  call cascade_all(lenlabels, lenlabels, labelsout1, labelsout)

end subroutine resolve_labels


subroutine relabel(group, lengroup, labels, lenlabels, groupout)

  ! Resolve group ID clashes.
  !
  ! Parameters
  ! ----------
  ! group : int array
  !   Group ID.
  ! lengroup : int
  !   Length of the group ID.
  ! labels : int array
  !   Label array.
  ! lenlabels : int
  !   Length of label array.
  !
  ! Returns
  ! -------
  ! groupout : int array
  !   Output group ID for each point.

  implicit none

  integer, intent(in) :: lengroup, lenlabels
  integer, intent(in) :: group(lengroup), labels(lenlabels)
  integer, intent(out) :: groupout(lengroup)

  integer :: i

  do i = 1, lengroup
    if (group(i) .ne. 0) then
      groupout(i) = labels(group(i))
    end if
  end do

end subroutine relabel


subroutine sum4group(group, param, lengroup, maxlabel, sumparam)

  ! Sum a parameter for each group ID.
  !
  ! Parameters
  ! ----------
  ! group : int array
  !   Group IDs.
  ! param : float array
  !   Parameter values for each point in the grid.
  ! lengroup : int
  !   Length of the group ID.
  ! maxlabel : int
  !   Maximum label.
  !
  ! Returns
  ! -------
  ! sumparam : int array
  !   Sum parameter values for each group.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lengroup, maxlabel
  integer, intent(in) :: group(lengroup)
  real(kind=dp), intent(in) :: param(lengroup)
  real(kind=dp), intent(out) :: sumparam(maxlabel)

  integer :: i

  do i = 1, maxlabel
    sumparam(i) = 0.
  end do

  do i = 1, lengroup
    if (group(i) .ne. 0) then
      sumparam(group(i)) = sumparam(group(i)) + param(i)
    end if
  end do

end subroutine sum4group


subroutine avg4group(group, param, lengroup, maxlabel, avgparam)

  ! Average for a parameter for each group ID.
  !
  ! Parameters
  ! ----------
  ! group : int array
  !   Group IDs.
  ! param : float array
  !   Parameter values for each point in the grid.
  ! lengroup : int
  !   Length of the group ID.
  ! maxlabel : int
  !   Maximum label.
  !
  ! Returns
  ! -------
  ! avgparam : int array
  !   Average parameter values for each group.

  implicit none
  integer, parameter :: dp = kind(1.d0)

  integer, intent(in) :: lengroup, maxlabel
  integer, intent(in) :: group(lengroup)
  real(kind=dp), intent(in) :: param(lengroup)
  real(kind=dp), intent(out) :: avgparam(maxlabel)

  integer :: i, nlabels(maxlabel)
  real(kind=dp) :: sumparam(maxlabel)

  call sum4group(group, param, lengroup, maxlabel, sumparam)
  call get_nlabels(maxlabel, lengroup, group, nlabels)

  do i = 1, maxlabel
    if (nlabels(i) .ne. 0) then
      avgparam(i) = sumparam(i)/real(nlabels(i))
    end if
  end do

end subroutine avg4group
