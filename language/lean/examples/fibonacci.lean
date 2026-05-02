-- Fibonacci with an iterative implementation and correctness proof.

def fib : Nat -> Nat
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

def fibIter (n : Nat) : Nat :=
  let rec loop (k a b : Nat) : Nat :=
    match k with
    | 0 => a
    | k + 1 => loop k b (a + b)
  loop n 0 1

theorem fibIter_loop_spec (k a b : Nat) :
    fibIter.loop k a b = a * fib k + b * fib (k + 1) := by
  induction k generalizing a b with
  | zero => simp [fibIter.loop, fib]
  | succ k ih =>
    simp [fibIter.loop, ih]
    simp [fib]
    ring

theorem fibIter_eq_fib (n : Nat) : fibIter n = fib n := by
  simp [fibIter, fibIter_loop_spec, fib]
