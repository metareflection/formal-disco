use vstd::prelude::*;

verus! {

// This program has an unprovable postcondition.
fn bad_abs(x: i64) -> (result: i64)
    ensures
        result >= 0,
{
    x  // wrong: doesn't handle negative x
}

} // verus!
