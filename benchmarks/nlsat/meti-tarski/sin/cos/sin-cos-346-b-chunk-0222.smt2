(set-logic QF_NRA)

(declare-fun skoSQ3 () Real)
(declare-fun pi () Real)
(declare-fun skoX () Real)
(assert (and (<= (* skoSQ3 (* skoSQ3 (/ (- 1.) 3.))) 0.) (and (= (* skoSQ3 skoSQ3) 3.) (and (not (<= (+ (/ (- 1.) 10000000.) (* pi (/ 1. 2.))) skoX)) (and (not (<= pi (/ 15707963. 5000000.))) (and (not (<= (/ 31415927. 10000000.) pi)) (and (not (<= skoX 0.)) (not (<= skoSQ3 0.)))))))))
(set-info :status sat)
(check-sat)