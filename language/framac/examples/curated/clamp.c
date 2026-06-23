/* Curated from acsl-by-example: MinMax/clamp.c
 * Self-contained: typedefs and helper predicates inlined.
 * Verifies with: frama-c -wp -wp-prover CVC5,Alt-Ergo
 */


/* === inlined: clamp.h === */

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
  requires   bound:  lower < upper;

  terminates         \true;
  exits              \false;
  assigns            \nothing;

  ensures    bound:  lower <= \result <= upper;

  behavior lower_bound:
    assumes          v < lower;
    ensures result:  \result == lower;

  behavior between:
    assumes          lower <= v <= upper;
    ensures result:  \result == v;

  behavior upper_bound:
    assumes          upper < v;
    ensures result:  \result == upper;

  complete behaviors;
  disjoint behaviors;
*/
value_type clamp(value_type v, value_type lower, value_type upper);

/* === inlined: LessThanComparable.acsl === */

/* === inlined: typedefs.h === */
/* (already inlined: typedefs.h) */

/*@
  lemma Less_Irreflexivity:
    \forall value_type a; !(a < a);

  lemma Less_Antisymmetry:
    \forall value_type a, b; (a < b)     ==>  !(b < a);

  lemma Less_Transitivity:
    \forall value_type a, b, c; (a < b)  ==>  (b < c)  ==>  (a < c);

  lemma Greater_Less:
    \forall value_type a, b;  (a > b)   <==>  (b < a);

  lemma LessOrEqual_Less:
    \forall value_type a, b;  (a <= b)  <==>  !(b < a);

  lemma GreaterOrEqual_Less:
    \forall value_type a, b;  (a >= b)  <==> !(a < b);
*/
value_type clamp(value_type v, value_type lower, value_type upper)
{
  return (v < lower) ? lower : (upper < v) ? upper : v;
}

