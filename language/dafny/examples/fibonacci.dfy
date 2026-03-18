// Functional definition and iterative computation of Fibonacci numbers.
function Fib(n: nat): nat {
  if n == 0 then 0
  else if n == 1 then 1
  else Fib(n - 1) + Fib(n - 2)
}

// Iterative implementation, proved equivalent to the functional spec.
method FibIter(n: nat) returns (result: nat)
  ensures result == Fib(n)
{
  if n == 0 { return 0; }
  var a := 0;
  var b := 1;
  var i := 1;
  while i < n
    invariant 1 <= i <= n
    invariant a == Fib(i - 1)
    invariant b == Fib(i)
    decreases n - i
  {
    var tmp := a + b;
    a := b;
    b := tmp;
    i := i + 1;
  }
  result := b;
}
