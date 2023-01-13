include "pixel_util.f90"
include "pixel_1dto2d.f90"
include "pixel_1dto3d.f90"


subroutine cascade(lenlabels, labels, indexin, indexout)

  implicit none

  integer, intent(in) :: lenlabels, labels(lenlabels), indexin
  integer, intent(out) :: indexout

  indexout = indexin

  do while (labels(indexout) .ne. indexout)
    indexout = labels(indexout)
  end do

end subroutine cascade


subroutine cascade_all(maxlabel, lenlabels, labels, labelsout)

  implicit none

  integer, intent(in) :: maxlabel, lenlabels, labels(lenlabels)
  integer, intent(out) :: labelsout(lenlabels)

  integer :: i, j

  do i = 1, maxlabel
    labelsout(i) = labels(i)
    call cascade(lenlabels, labels, labels(i), j)
    labelsout(i) = j
  end do

end subroutine cascade_all


subroutine unionise(ind1, ind2, lenlabels, labels, ind1out, ind2out, indout)

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

  implicit none

  integer, intent(in) :: maxlabel, lenlabels
  integer, intent(in) :: labels(lenlabels)
  integer, intent(out) :: nlabels(maxlabel)

  integer :: i

  do i = 1, maxlabel
    nlabels(i) = 0
  end do

  do i = 1, lenlabels
    nlabels(labels(i)) = nlabels(labels(i)) + 1
  end do

end subroutine get_nlabels


subroutine shuffle_down(maxlabel, lenlabels, labels, newmaxlabel, labelsout)

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


subroutine hoshen_kopelman_2d(binmap, nxgrid, nygrid, periodx, periody, maxlabel, group)

  implicit none

  integer, intent(in) :: nxgrid, nygrid
  integer, intent(in) :: binmap(nxgrid*nygrid)
  logical, intent(in) :: periodx, periody
  integer, intent(out) :: maxlabel, group(nxgrid*nygrid)

  integer :: i, j, ix, iy, jx, jy, xpix_id, ypix_id
  integer :: newmaxlabel, labels(nxgrid*nygrid)
  integer :: xpix_plus_id, ypix_plus_id
  integer :: i_px, i_py
  integer :: ind1out, ind2out, indout
  integer :: labelsout1(nxgrid*nygrid)

  maxlabel = 0

  do i = 1, nxgrid*nygrid
    group(i) = 0
  end do

  do i = 1, nxgrid*nygrid
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
            call unionise(group(i), group(i_px), nxgrid*nygrid, labels, ind1out, ind2out, indout)
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
            call unionise(group(i), group(i_py), nxgrid*nygrid, labels, ind1out, ind2out, indout)
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

  call cascade_all(maxlabel, nxgrid*nygrid, labels, labelsout1)
  call shuffle_down(maxlabel, nxgrid*nygrid, labelsout1, newmaxlabel, labels)

  do i = 1, nxgrid*nygrid
    if (group(i) .ne. 0) then
      group(i) = labels(group(i))
    end if
  end do

  maxlabel = newmaxlabel

end subroutine hoshen_kopelman_2d

subroutine hoshen_kopelman_3d(binmap, nxgrid, nygrid, nzgrid, periodx, periody &
  , periodz, maxlabel, group)

  implicit none

  integer, intent(in) :: nxgrid, nygrid, nzgrid
  integer, intent(in) :: binmap(nxgrid*nygrid*nzgrid)
  logical, intent(in) :: periodx, periody, periodz
  integer, intent(out) :: maxlabel, group(nxgrid*nygrid*nzgrid)

  integer :: i, j, ix, iy, iz, jx, jy, jz, xpix_id, ypix_id, zpix_id
  integer :: newmaxlabel, labels(nxgrid*nygrid*nzgrid)
  integer :: xpix_plus_id, ypix_plus_id, zpix_plus_id
  integer :: i_px, i_py, i_pz
  integer :: ind1out, ind2out, indout
  integer :: labelsout1(nxgrid*nygrid*nzgrid)

  maxlabel = 0

  do i = 1, nxgrid*nygrid*nzgrid
    group(i) = 0
  end do

  do i = 1, nxgrid*nygrid*nzgrid
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
            call unionise(group(i), group(i_px), nxgrid*nygrid*nzgrid, labels, ind1out, ind2out, indout)
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
            call unionise(group(i), group(i_py), nxgrid*nygrid*nzgrid, labels, ind1out, ind2out, indout)
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
            call unionise(group(i), group(i_pz), nxgrid*nygrid*nzgrid, labels, ind1out, ind2out, indout)
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

  call cascade_all(maxlabel, nxgrid*nygrid*nzgrid, labels, labelsout1)
  call shuffle_down(maxlabel, nxgrid*nygrid*nzgrid, labelsout1, newmaxlabel, labels)

  do i = 1, nxgrid*nygrid*nzgrid
    if (group(i) .ne. 0) then
      group(i) = labels(group(i))
    end if
  end do

  maxlabel = newmaxlabel

end subroutine hoshen_kopelman_3d
