/*@
  requires n >= 0;
  requires \valid(a + (0 .. n-1));
  assigns \nothing;
  ensures \result == \sum(0, n-1, \lambda integer i; a[i]);
*/
long long sum_array(int *a, int n) {
  long long s = 0;
  /*@
    loop invariant 0 <= i <= n;
    loop invariant s == \sum(0, i-1, \lambda integer k; a[k]);
    loop assigns i, s;
    loop variant n - i;
  */
  for (int i = 0; i < n; i++) {
    s += a[i];
  }
  return s;
}
