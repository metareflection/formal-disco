use vstd::prelude::*;

verus! {

fn binary_search(a: &Vec<i64>, target: i64) -> (index: usize)
    requires
        forall|i: int, j: int| 0 <= i < j < a.len() ==> a[i] <= a[j],
    ensures
        index < a.len() ==> a[index as int] == target,
        index == a.len() ==> forall|k: int| 0 <= k < a.len() ==> a[k] != target,
{
    let mut lo: usize = 0;
    let mut hi: usize = a.len();

    while lo < hi
        invariant
            0 <= lo <= hi <= a.len(),
            forall|k: int| 0 <= k < lo as int ==> a[k] != target,
            forall|k: int| hi as int <= k < a.len() ==> a[k] != target,
            forall|i: int, j: int| 0 <= i < j < a.len() ==> a[i] <= a[j],
        decreases hi - lo,
    {
        let mid = lo + (hi - lo) / 2;
        if a[mid] == target {
            return mid;
        } else if a[mid] < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    a.len()
}

} // verus!
