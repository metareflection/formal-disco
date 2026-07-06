/* Curated from acsl-by-example: Numeric/iota.c
 * Self-contained: typedefs and helper predicates inlined.
 * Verifies with: frama-c -wp -wp-prover CVC5,Alt-Ergo
 */


/* === inlined: iota.h === */

/* === inlined: IotaGenerate.acsl === */

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
  predicate IotaGenerate(value_type* a, integer n, value_type v) =
    \forall integer i; 0 <= i < n  ==>  a[i] == v+i;
*/
/* unresolved include: limits.h */
/*@
  requires   valid:      \valid(a + (0..n-1));
  requires   limit:      v + n <= VALUE_TYPE_MAX;

  terminates             \true;
  exits                  \false;
  assigns                a[0..n-1];

  ensures    increment:  IotaGenerate(a, n, v);
*/
void iota(value_type* a, size_type n, value_type v);
void iota(value_type* a, size_type n, value_type v)
{
  /*@
    loop invariant bound:     0 <= i <= n;
    loop invariant limit:     v == \at(v, Pre) + i;
    loop invariant increment: IotaGenerate(a, i, \at(v, Pre));

    loop assigns i, v, a[0..n-1];
    loop variant n-i;
  */
  for (size_type i = 0u; i < n; ++i) {
    a[i] = v++;
  }
}
