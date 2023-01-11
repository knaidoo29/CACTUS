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


subroutine unionise(ind1, ind2, maxlabel, lenlabels, labels, labelsout)

  implicit none

  integer, intent(in) :: ind1, ind2, maxlabel, lenlabels
  integer, intent(in) :: labels(lenlabels)
  integer, intent(out) :: labelsout(lenlabels)

  integer :: i, ind1_, ind2_, ind_

  do i = 1, maxlabel
    labelsout(i) = labels(i)
  end do

  call cascade(lenlabels, labels, ind1, ind1_)
  call cascade(lenlabels, labels, ind2, ind2_)

  if (ind1_ .le. ind2_) then
    ind_ = ind1_
  else
    ind_ = ind2_
  end if

  labelsout(ind1_) = ind_
  labelsout(ind2_) = ind_

  labelsout(ind1) = ind_
  labelsout(ind2) = ind_

end subroutine unionise


subroutine get_nlabels(maxlabel, lenlabels, labels, nlabels)

  implicit none

  integer, intent(in) :: maxlabel, lenlabels
  integer, intent(in) :: labels(lenlabels)
  integer, intent(out) :: nlabels(maxlabel)

  integer :: i

  do i = 1, maxlabel
    nlabels(labels(i)) = nlabels(labels(i)) + 1
  end do

end subroutine get_nlabels


subroutine remove_label_gaps(maxlabel, lenlabels, labels, newmaxlabel, labelsout)

  implicit none

  integer, intent(in) :: maxlabel, lenlabels
  integer, intent(in) :: labels(lenlabels)
  integer, intent(out) :: newmaxlabel, labelsout(lenlabels)

  integer :: i, nlabels(maxlabel), j, maplabel(maxlabel)

  call get_nlabels(maxlabel, lenlabels, labels, nlabels)

  j = 0
  do i = 1, maxlabel
    if (nlabels(i) .ne. 0) then
      j = j + 1
      maplabel(i) = j
    end if
  end do

  newmaxlabel = j

  do i = 1, lenlabels
    labelsout(i) = maplabel(labels(i))
  end do

end subroutine remove_label_gaps
