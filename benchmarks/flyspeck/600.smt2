(set-logic QF_NRA)
(declare-fun x1 () Real)
(declare-fun x2 () Real)
(declare-fun x3 () Real)
(declare-fun x4 () Real)
(declare-fun x5 () Real)
(declare-fun x6 () Real)
(assert (<= 2.5854 x1))
(assert (<= x1 2.6181))
(assert (<= 2.0 x2))
(assert (<= x2 2.46350884418))
(assert (<= 2.0 x3))
(assert (<= x3 2.46350884418))
(assert (<= 2.0 x4))
(assert (<= x4 2.46350884418))
(assert (<= 2.0 x5))
(assert (<= x5 2.46350884418))
(assert (<= 1.0 x6))
(assert (<= x6 1.0))
(assert (not (< (+ (* 1.0 (* 2.0 (* 3.14159265 (- 0.0469)))) (+ (* x1 0.051237) (+ (* x1 (- 0.02241)) (+ (* x1 (- 0.014413)) (+ (* x1 (- 0.014413)) (+ (* x2 (* 2.0 0.060747)) (+ (* x2 (* 2.0 (- 0.060747))) (+ (* x3 (* 2.0 (- 0.060747))) (+ (* x3 (* 2.0 0.060747)) (+ (* x4 (* 2.0 0.060747)) (+ (* x4 (* 2.0 (- 0.060747))) (+ (* x5 (* 2.0 (- 0.060747))) (+ (* x5 (* 2.0 0.060747)) (+ (* 1.0 (- 0.530637)) (+ (* 1.0 0.608509) (+ (* 1.0 (- 0.377571)) (* 1.0 0.594377))))))))))))))))) 0.0)))
(check-sat)
(exit)
