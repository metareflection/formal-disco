/*@
  requires n > 0;
  requires \valid(a + (0 .. n-1));
  requires \forall integer i, j; 0 <= i <= j < n ==> a[i] <= a[j];
  assigns \nothing;
  ensures -1 <= \result < n;
  ensures \result >= 0 ==> a[\result] == v;
  ensures \result == -1 ==> \forall integer i; 0 <= i < n ==> a[i] != v;
*/
int binary_search(int *a, int n, int v) {
  int lo = 0, hi = n;
  /*@
    loop invariant 0 <= lo <= hi <= n;
    loop invariant \forall integer i; 0 <= i < lo ==> a[i] != v;
    loop invariant \forall integer i; hi <= i < n ==> a[i] != v;
    loop assigns lo, hi;
    loop variant hi - lo;
  */
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (a[mid] == v) return mid;
    else if (a[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return -1;
}
