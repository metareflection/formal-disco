// Binary search over a sorted array.
// Returns an index where target is found, or -1 if not present.
method BinarySearch(a: array<int>, target: int) returns (index: int)
  requires forall i, j :: 0 <= i < j < a.Length ==> a[i] <= a[j]
  ensures index == -1 || (0 <= index < a.Length && a[index] == target)
  ensures index == -1 ==> forall k :: 0 <= k < a.Length ==> a[k] != target
{
  var lo := 0;
  var hi := a.Length;
  index := -1;

  while lo < hi
    invariant 0 <= lo <= hi <= a.Length
    invariant forall k :: 0 <= k < lo ==> a[k] != target
    invariant forall k :: hi <= k < a.Length ==> a[k] != target
    decreases hi - lo
  {
    var mid := lo + (hi - lo) / 2;
    if a[mid] == target {
      index := mid;
      return;
    } else if a[mid] < target {
      lo := mid + 1;
    } else {
      hi := mid;
    }
  }
}
