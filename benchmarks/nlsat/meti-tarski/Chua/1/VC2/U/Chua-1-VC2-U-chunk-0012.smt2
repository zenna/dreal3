(set-logic QF_NRA)

(declare-fun skoX () Real)
(declare-fun skoS () Real)
(declare-fun skoC () Real)
(assert (and (not (<= (* skoC (/ 3. 13.)) skoS)) (and (<= skoX 289.) (<= 0. skoX))))
(set-info :status sat)
(check-sat)