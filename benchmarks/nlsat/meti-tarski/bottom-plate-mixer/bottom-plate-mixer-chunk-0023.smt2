(set-logic QF_NRA)

(declare-fun skoX () Real)
(declare-fun skoS () Real)
(declare-fun skoC () Real)
(declare-fun skoCB () Real)
(declare-fun skoSB () Real)
(assert (and (<= (/ 177. 366500000.) skoX) (and (<= (+ (/ 760000. 7383.) (* skoC (/ (- 3400.) 7383.))) skoS) (and (<= skoS (+ (/ 760000. 7383.) (* skoC (/ (- 3400.) 7383.)))) (and (<= skoX (/ 177. 366500000.)) (and (not (<= skoSB (+ (+ (+ (/ 12695. 52.) (* skoC (/ (- 570.) 13.))) (* skoCB (/ (- 49.) 65.))) (* skoS (- 200.))))) (and (<= skoX (/ 1. 10000000.)) (<= 0. skoX))))))))
(set-info :status unsat)
(check-sat)