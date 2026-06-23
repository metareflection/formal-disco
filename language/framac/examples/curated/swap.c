/* Curated from acsl-by-example: Mutating/swap.c
 * Self-contained: typedefs and helper predicates inlined.
 * Verifies with: frama-c -wp -wp-prover CVC5,Alt-Ergo
 */


/* === inlined: swap.h === */

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
  requires   valid:     \valid(p);
  requires   valid:     \valid(q);

  terminates            \true;
  exits                 \false;
  assigns               *p, *q;

  ensures    exchange:  *p == \old(*q);
  ensures    exchange:  *q == \old(*p);
*/
void swap(value_type* p, value_type* q);
void swap(value_type* p, value_type* q)
{
  value_type save = *p;
  *p = *q;
  *q = save;
}

