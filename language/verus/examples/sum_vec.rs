use vstd::prelude::*;

verus! {

spec fn sum_spec(s: Seq<i64>, i: int) -> int
    decreases i,
{
    if i <= 0 {
        0
    } else {
        sum_spec(s, i - 1) + s[i - 1]
    }
}

proof fn sum_spec_bound(s: Seq<i64>, i: int)
    requires
        0 <= i <= s.len(),
        forall|j: int| 0 <= j < s.len() ==> 0 <= #[trigger] s[j] <= 1000,
        s.len() < 1000,
    ensures
        0 <= sum_spec(s, i) <= i * 1000,
    decreases i,
{
    if i > 0 {
        sum_spec_bound(s, i - 1);
    }
}

fn sum_vec(v: &Vec<i64>) -> (s: i64)
    requires
        v.len() < 1000,
        forall|i: int| 0 <= i < v.len() ==> 0 <= #[trigger] v[i] <= 1000,
    ensures
        s == sum_spec(v@, v.len() as int),
{
    let mut s: i64 = 0;
    let mut i: usize = 0;

    while i < v.len()
        invariant
            0 <= i <= v.len(),
            s == sum_spec(v@, i as int),
            0 <= s <= i as int * 1000,
            v.len() < 1000,
            forall|j: int| 0 <= j < v.len() ==> 0 <= #[trigger] v[j] <= 1000,
        decreases v.len() - i,
    {
        proof { sum_spec_bound(v@, i as int); }
        s = s + v[i];
        i = i + 1;
    }
    s
}

} // verus!
