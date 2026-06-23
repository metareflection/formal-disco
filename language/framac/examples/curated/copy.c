/* Curated from acsl-by-example: Mutating/copy.c
 * Self-contained: typedefs and helper predicates inlined.
 * Verifies with: frama-c -wp -wp-prover CVC5,Alt-Ergo
 */


/* === inlined: copy.h === */

/* === inlined: Equal.acsl === */

/* === inlined: typedefs.h === */
/* Replaces <limits.h> for self-containment. */
#define INT_MAX    2147483647
#define INT_MIN  (-2147483647 - 1)
#define UINT_MAX 4294967295U

#ifndef __cplusplus
typedef int bool;
#define false		((bool)0)
#define true		((bool)1)
#endif

typedef int value_type;

#define VALUE_TYPE_MAX  INT_MAX
#define VALUE_TYPE_MIN  INT_MIN

typedef unsigned int size_type;

#define SIZE_TYPE_MAX  UINT_MAX
/*@
  predicate Equal{K,L}(value_type* a, integer m, integer n, value_type* b) =
    \forall integer i; m <= i < n  ==>  \at(a[i],K) == \at(b[i],L);

  predicate Equal{K,L}(value_type* a, integer n, value_type* b) =
    Equal{K,L}(a, 0, n, b);

  predicate Equal{K,L}(value_type* a, integer m, integer n,
                       value_type* b, integer p) =
    \forall integer k; 0 <= k < n-m ==> \at(a[m+k],K) == \at(b[p+k],L);

  predicate Equal{K,L}(value_type* a, integer m, integer n, integer p) =
      Equal{K,L}(a, m, n, a, p);
*/
/*@
  requires   valid:  \valid_read(a + (0..n-1));
  requires   valid:  \valid(b + (0..n-1));
  requires   sep:    \separated(a + (0..n-1), b);

  terminates         \true;
  exits              \false;
  assigns            b[0..n-1];

  ensures    equal:  Equal{Old,Here}(a, n, b);
*/
void copy(const value_type* a, const size_type n, value_type* b);

/* === inlined: Unchanged.acsl === */

/* === inlined: typedefs.h === */
/* (already inlined: typedefs.h) */

/*@
  predicate Unchanged{K,L}(value_type* a, integer m, integer n) =
    \forall integer i; m <= i < n ==>  \at(a[i],K) == \at(a[i],L);

  predicate Unchanged{K,L}(value_type* a, integer n) = Unchanged{K,L}(a, 0, n);
*/
void copy(const value_type* a, size_type n, value_type* b)
{
  /*@
    loop invariant bound:     0 <= i <= n;
    loop invariant equal:     Equal{Pre,Here}(a, i, b);
    loop invariant unchanged: Unchanged{Pre,Here}(a, i, n);
    loop assigns   i, b[0..n-1];
    loop variant n-i;
  */
  for (size_type i = 0u; i < n; ++i) {
    b[i] = a[i];
  }
}

