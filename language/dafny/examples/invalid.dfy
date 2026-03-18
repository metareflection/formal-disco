// This program has an incorrect postcondition that Dafny cannot prove.
method Add(a: int, b: int) returns (result: int)
  ensures result == a + b + 1  // off by one — wrong on purpose
{
  result := a + b;
}
