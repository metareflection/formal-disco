// This program has an unprovable postcondition.
/*@ ensures \result >= 0; */
int bad_abs(int x) {
  return x;  // wrong: doesn't handle negative x
}
