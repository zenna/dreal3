(set-logic QF_NRA)

(declare-fun skoSX () Real)
(declare-fun skoSMX () Real)
(declare-fun skoX () Real)
(assert (and (<= (+ (- 4.) (* skoSMX (- 1.))) skoSX) (and (not (<= skoX 0.)) (and (<= 0. skoSMX) (and (<= 0. skoSX) (and (<= skoX 1.) (and (= (+ (- 1.) (* skoSX skoSX)) skoX) (= skoX (+ 1. (* skoSMX (* skoSMX (- 1.))))))))))))
(set-info :status sat)
(check-sat)