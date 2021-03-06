(set-logic QF_NRA_ODE)
(declare-fun v () Real)
(declare-fun v_0_0 () Real)
(declare-fun v_0_t () Real)
(declare-fun x () Real)
(declare-fun x_0_0 () Real)
(declare-fun x_0_t () Real)
(declare-fun y () Real)
(declare-fun y_0_0 () Real)
(declare-fun y_0_t () Real)
(declare-fun z () Real)
(declare-fun z_0_0 () Real)
(declare-fun z_0_t () Real)
(declare-fun alphay () Real)
(declare-fun alphay_0_0 () Real)
(declare-fun alphay_0_t () Real)
(declare-fun betax () Real)
(declare-fun betax_0_0 () Real)
(declare-fun betax_0_t () Real)
(declare-fun t () Real)
(declare-fun t_0_0 () Real)
(declare-fun t_0_t () Real)
(declare-fun time_0 () Real)
(define-ode flow_1
  ((= d/dt[alphay] 0.0)
   (= d/dt[betax] 0.0)
   (= d/dt[t] 1.0)
   (= d/dt[v] (+ (* (- (- (- (/ 0.0197 (+ 1.0 (exp (* (- 10.0 z) 1.0)))) (/ betax (+ 1.0 (exp (* (- z 10.0) 2.0))))) (* 5.0E-5 (- 1.0 (/ z 12.0)))) 0.01) x) (+ 0.02 (+ (* 5.0E-5 (* (- 1.0 (/ z 12.0)) x)) (* (- (* alphay (- 1.0 (* 1.0 (/ z 12.0)))) 0.0168) y)))))
   (= d/dt[x] (+ (* (- (- (- (/ 0.0197 (+ 1.0 (exp (* (- 10.0 z) 1.0)))) (/ betax (+ 1.0 (exp (* (- z 10.0) 2.0))))) (* 5.0E-5 (- 1.0 (/ z 12.0)))) 0.01) x) 0.02))
   (= d/dt[y] (+ (* 5.0E-5 (* (- 1.0 (/ z 12.0)) x)) (* (- (* alphay (- 1.0 (* 1.0 (/ z 12.0)))) 0.0168) y)))
   (= d/dt[z] (+ (* (- 0.0 z) 0.08) 0.03))))
(assert (>= v_0_0 0))
(assert (<= v_0_0 100))
(assert (>= v_0_t 0))
(assert (<= v_0_t 100))
(assert (>= x_0_0 0))
(assert (<= x_0_0 100))
(assert (>= x_0_t 0))
(assert (<= x_0_t 100))
(assert (>= y_0_0 0))
(assert (<= y_0_0 100))
(assert (>= y_0_t 0))
(assert (<= y_0_t 100))
(assert (>= z_0_0 0))
(assert (<= z_0_0 100))
(assert (>= z_0_t 0))
(assert (<= z_0_t 100))
(assert (>= alphay_0_0 0))
(assert (<= alphay_0_0 0.025))
(assert (>= alphay_0_t 0))
(assert (<= alphay_0_t 0.025))
(assert (>= betax_0_0 0))
(assert (<= betax_0_0 0.025))
(assert (>= betax_0_t 0))
(assert (<= betax_0_t 0.025))
(assert (>= time_0 0))
(assert (<= time_0 83))
(assert (>= t_0_0 0))
(assert (<= t_0_0 83))
(assert (>= t_0_t 0))
(assert (<= t_0_t 83))
(assert
(and
(= [alphay_0_t betax_0_t t_0_t v_0_t x_0_t y_0_t z_0_t] (integral 0. time_0 [alphay_0_0 betax_0_0 t_0_0 v_0_0 x_0_0 y_0_0 z_0_0] flow_1))
(= alphay_0_0 alphay_0_t)
(= betax_0_0 betax_0_t)
(= t_0_0 0)
(>= v_0_0 19.0998)
(<= v_0_0 19.1002)
(>= x_0_0 18.9998)
(<= x_0_0 19.0002)
(>= y_0_0 0.099999)
(<= y_0_0 0.100001)
(>= z_0_0 12.4999)
(<= z_0_0 12.5001)
(= t_0_t 83)
(>= v_0_t 1.1)
(<= v_0_t 3.9)
)
)
(check-sat)
(exit)
