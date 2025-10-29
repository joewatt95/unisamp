; SMT-LIB file to compute kappa given epsilon
;
; Original formula:
; epsilon = (1 + kappa) * (2.23 + 0.48 / (1 - kappa)^2) - 1
;
; We are solving for 'kappa' given a value for 'epsilon'.

; Set the logic to Quantifier-Free Non-Linear Real Arithmetic
(set-logic QF_NRA)

; Declare the constants (variables) we will use
(declare-const epsilon Real)
(declare-const kappa Real)


; --- USER INPUT ---
; Set the value for epsilon here.
; For this example, we'll use epsilon = 0.5
(assert (= epsilon 6.83201))
; ------------------


; --- CONSTRAINTS ---

; 1. Assert the main formula.
; SMT-LIB uses prefix notation: (operator operand1 operand2)
(assert
  (= epsilon
     (-
       (* (+ 1.0 kappa)
          (+ 7.44 (/ 0.392 (* (- 1.0 kappa) (- 1.0 kappa))))
       )
       1.0
     )
  )
)

; 2. Add physical/mathematical constraints for kappa.
; The formula is undefined at kappa = 1, so we must exclude it.
; In many physical models, kappa is also bounded between 0 and 1.
; Adding these bounds helps the solver find the unique, physically
; meaningful solution much faster.
(assert (> kappa 0.0))
(assert (< kappa 1.0))


; --- SOLVER COMMANDS ---

; Check if a solution exists under these constraints
(check-sat)

; If a solution exists (sat), get the value of kappa
(get-value (kappa))
