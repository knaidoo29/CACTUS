
subroutine pix_f2p(pixin, pixout)

  implicit none

  integer, intent(in) :: pixin
  integer, intent(out) :: pixout

  pixout = pixin - 1

end subroutine pix_f2p


subroutine pix_p2f(pixin, pixout)

  implicit none

  integer, intent(in) :: pixin
  integer, intent(out) :: pixout

  pixout = pixin + 1

end subroutine pix_p2f
