(set-logic QF_NRA)

(declare-fun skoX () Real)
(declare-fun skoZ () Real)
(declare-fun skoY () Real)
(assert (and (<= (+ (+ 1. (* skoX (- 1.))) (* skoY (- 1.))) skoZ) (and (<= skoX (/ (- 1.) 2.)) (and (not (<= skoZ (/ 1. 20.))) (and (not (<= skoY (/ 1. 20.))) (and (not (<= skoX (/ 1. 20.))) (and (not (<= 15. skoZ)) (and (not (<= 15. skoY)) (not (<= 15. skoX))))))))))
(set-info :status unsat)
(check-sat)