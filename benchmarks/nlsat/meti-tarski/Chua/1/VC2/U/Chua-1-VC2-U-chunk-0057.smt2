(set-logic QF_NRA)

(declare-fun skoS () Real)
(declare-fun skoC () Real)
(declare-fun skoX () Real)
(assert (and (not (<= 0. skoX)) (not (<= (* skoC (/ 3. 13.)) skoS))))
(set-info :status sat)
(check-sat)