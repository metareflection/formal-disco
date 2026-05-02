-- Simple max function with a proof of correctness.

def max' (a b : Nat) : Nat :=
  if a >= b then a else b

theorem max'_ge_left (a b : Nat) : max' a b >= a := by
  unfold max'
  split <;> omega

theorem max'_ge_right (a b : Nat) : max' a b >= b := by
  unfold max'
  split <;> omega

theorem max'_eq_or (a b : Nat) : max' a b = a \/ max' a b = b := by
  unfold max'
  split
  · left; rfl
  · right; rfl
