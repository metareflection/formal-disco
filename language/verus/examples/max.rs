use vstd::prelude::*;

verus! {

fn max(a: i64, b: i64) -> (result: i64)
    ensures
        result >= a,
        result >= b,
        result == a || result == b,
{
    if a >= b {
        a
    } else {
        b
    }
}

} // verus!
