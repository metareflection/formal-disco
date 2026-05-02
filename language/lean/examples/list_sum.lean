-- List sum with correctness properties.

def listSum : List Nat -> Nat
  | [] => 0
  | x :: xs => x + listSum xs

theorem listSum_nil : listSum [] = 0 := rfl

theorem listSum_cons (x : Nat) (xs : List Nat) :
    listSum (x :: xs) = x + listSum xs := rfl

theorem listSum_append (xs ys : List Nat) :
    listSum (xs ++ ys) = listSum xs + listSum ys := by
  induction xs with
  | nil => simp [listSum]
  | cons x xs ih =>
    simp [listSum, ih]
    omega

theorem listSum_replicate (n k : Nat) :
    listSum (List.replicate n k) = n * k := by
  induction n with
  | zero => simp [listSum]
  | succ n ih =>
    simp [List.replicate, listSum, ih]
    ring
