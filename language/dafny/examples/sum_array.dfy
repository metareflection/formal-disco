// Computes the sum of all elements in an integer array.
function SumTo(a: array<int>, n: int): int
  requires 0 <= n <= a.Length
  reads a
{
  if n == 0 then 0 else SumTo(a, n - 1) + a[n - 1]
}

method SumArray(a: array<int>) returns (total: int)
  ensures total == SumTo(a, a.Length)
{
  total := 0;
  var i := 0;

  while i < a.Length
    invariant 0 <= i <= a.Length
    invariant total == SumTo(a, i)
    decreases a.Length - i
  {
    total := total + a[i];
    i := i + 1;
  }
}
