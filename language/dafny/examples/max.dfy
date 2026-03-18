// Returns the maximum of two integers.
method Max(a: int, b: int) returns (result: int)
  ensures result >= a && result >= b
  ensures result == a || result == b
{
  if a >= b {
    result := a;
  } else {
    result := b;
  }
}
