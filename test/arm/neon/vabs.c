/* Copyright (c) 2019 Evan Nemerson <evan@nemerson.com>
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#define SIMDE_TESTS_CURRENT_NEON_OP abs
#include <test/arm/neon/test-neon-internal.h>
#include <simde/arm/neon.h>

static MunitResult
test_simde_vabs_s8(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int8x8_t a;
    simde_int8x8_t r;
  } test_vec[8] = {
    { simde_x_vload_s8(INT8_C( 118), INT8_C( 102), INT8_C(-111), INT8_C(   6),
                       INT8_C(  76), INT8_C( -43), INT8_C( -97), INT8_C(-121)),
      simde_x_vload_s8(INT8_C( 118), INT8_C( 102), INT8_C( 111), INT8_C(   6),
                       INT8_C(  76), INT8_C(  43), INT8_C(  97), INT8_C( 121)) },
    { simde_x_vload_s8(INT8_C(   0), INT8_C( -73), INT8_C(  68), INT8_C( -33),
                       INT8_C(  18), INT8_C( -92), INT8_C(   4), INT8_C( -34)),
      simde_x_vload_s8(INT8_C(   0), INT8_C(  73), INT8_C(  68), INT8_C(  33),
                       INT8_C(  18), INT8_C(  92), INT8_C(   4), INT8_C(  34)) },
    { simde_x_vload_s8(INT8_C( 113), INT8_C(  42), INT8_C( -84), INT8_C( 122),
                       INT8_C(  63), INT8_C(  51), INT8_C( -42), INT8_C(-108)),
      simde_x_vload_s8(INT8_C( 113), INT8_C(  42), INT8_C(  84), INT8_C( 122),
                       INT8_C(  63), INT8_C(  51), INT8_C(  42), INT8_C( 108)) },
    { simde_x_vload_s8(INT8_C(  74), INT8_C( -65), INT8_C( -76), INT8_C( -78),
                       INT8_C(  15), INT8_C( 118), INT8_C(  70), INT8_C(-112)),
      simde_x_vload_s8(INT8_C(  74), INT8_C(  65), INT8_C(  76), INT8_C(  78),
                       INT8_C(  15), INT8_C( 118), INT8_C(  70), INT8_C( 112)) },
    { simde_x_vload_s8(INT8_C(  28), INT8_C(  -8), INT8_C(  91), INT8_C( -95),
                       INT8_C(  80), INT8_C( -63), INT8_C(-101), INT8_C(  19)),
      simde_x_vload_s8(INT8_C(  28), INT8_C(   8), INT8_C(  91), INT8_C(  95),
                       INT8_C(  80), INT8_C(  63), INT8_C( 101), INT8_C(  19)) },
    { simde_x_vload_s8(INT8_C( -71), INT8_C( -86), INT8_C(  -3), INT8_C( -93),
                       INT8_C( -31), INT8_C(-112), INT8_C( -96), INT8_C( -17)),
      simde_x_vload_s8(INT8_C(  71), INT8_C(  86), INT8_C(   3), INT8_C(  93),
                       INT8_C(  31), INT8_C( 112), INT8_C(  96), INT8_C(  17)) },
    { simde_x_vload_s8(INT8_C( -45), INT8_C( -96), INT8_C( -67), INT8_C(  88),
                       INT8_C(-126), INT8_C(-122), INT8_C( -35), INT8_C( -99)),
      simde_x_vload_s8(INT8_C(  45), INT8_C(  96), INT8_C(  67), INT8_C(  88),
                       INT8_C( 126), INT8_C( 122), INT8_C(  35), INT8_C(  99)) },
    { simde_x_vload_s8(INT8_C( -56), INT8_C( -57), INT8_C(  12), INT8_C( -85),
                       INT8_C( -68), INT8_C(  -2), INT8_C(  47), INT8_C(-126)),
      simde_x_vload_s8(INT8_C(  56), INT8_C(  57), INT8_C(  12), INT8_C(  85),
                       INT8_C(  68), INT8_C(   2), INT8_C(  47), INT8_C( 126)) },

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int8x8_t r = simde_vabs_s8(test_vec[i].a);
    simde_neon_assert_int8x8(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabs_s16(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int16x4_t a;
    simde_int16x4_t r;
  } test_vec[8] = {
    { simde_x_vload_s16(INT16_C( 26230), INT16_C(  1681), INT16_C(-10932), INT16_C(-30817)),
      simde_x_vload_s16(INT16_C( 26230), INT16_C(  1681), INT16_C( 10932), INT16_C( 30817)) },
    { simde_x_vload_s16(INT16_C(-18688), INT16_C( -8380), INT16_C(-23534), INT16_C( -8700)),
      simde_x_vload_s16(INT16_C( 18688), INT16_C(  8380), INT16_C( 23534), INT16_C(  8700)) },
    { simde_x_vload_s16(INT16_C( 10865), INT16_C( 31404), INT16_C( 13119), INT16_C(-27434)),
      simde_x_vload_s16(INT16_C( 10865), INT16_C( 31404), INT16_C( 13119), INT16_C( 27434)) },
    { simde_x_vload_s16(INT16_C(-16566), INT16_C(-19788), INT16_C( 30223), INT16_C(-28602)),
      simde_x_vload_s16(INT16_C( 16566), INT16_C( 19788), INT16_C( 30223), INT16_C( 28602)) },
    { simde_x_vload_s16(INT16_C( -2020), INT16_C(-24229), INT16_C(-16048), INT16_C(  5019)),
      simde_x_vload_s16(INT16_C(  2020), INT16_C( 24229), INT16_C( 16048), INT16_C(  5019)) },
    { simde_x_vload_s16(INT16_C(-21831), INT16_C(-23555), INT16_C(-28447), INT16_C( -4192)),
      simde_x_vload_s16(INT16_C( 21831), INT16_C( 23555), INT16_C( 28447), INT16_C(  4192)) },
    { simde_x_vload_s16(INT16_C(-24365), INT16_C( 22717), INT16_C(-31102), INT16_C(-25123)),
      simde_x_vload_s16(INT16_C( 24365), INT16_C( 22717), INT16_C( 31102), INT16_C( 25123)) },
    { simde_x_vload_s16(INT16_C(-14392), INT16_C(-21748), INT16_C(  -324), INT16_C(-32209)),
      simde_x_vload_s16(INT16_C( 14392), INT16_C( 21748), INT16_C(   324), INT16_C( 32209)) },

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int16x4_t r = simde_vabs_s16(test_vec[i].a);
    simde_neon_assert_int16x4(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabs_s32(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int32x2_t a;
    simde_int32x2_t r;
  } test_vec[8] = {
    { simde_x_vload_s32(INT32_C(  110192246), INT32_C(-2019568308)),
      simde_x_vload_s32(INT32_C(  110192246), INT32_C( 2019568308)) },
    { simde_x_vload_s32(INT32_C( -549144832), INT32_C( -570121198)),
      simde_x_vload_s32(INT32_C(  549144832), INT32_C(  570121198)) },
    { simde_x_vload_s32(INT32_C( 2058103409), INT32_C(-1797901505)),
      simde_x_vload_s32(INT32_C( 2058103409), INT32_C( 1797901505)) },
    { simde_x_vload_s32(INT32_C(-1296777398), INT32_C(-1874430449)),
      simde_x_vload_s32(INT32_C( 1296777398), INT32_C( 1874430449)) },
    { simde_x_vload_s32(INT32_C(-1587808228), INT32_C(  328974672)),
      simde_x_vload_s32(INT32_C( 1587808228), INT32_C(  328974672)) },
    { simde_x_vload_s32(INT32_C(-1543656775), INT32_C( -274689823)),
      simde_x_vload_s32(INT32_C( 1543656775), INT32_C(  274689823)) },
    { simde_x_vload_s32(INT32_C( 1488822483), INT32_C(-1646426494)),
      simde_x_vload_s32(INT32_C( 1488822483), INT32_C( 1646426494)) },
    { simde_x_vload_s32(INT32_C(-1425225784), INT32_C(-2110783812)),
      simde_x_vload_s32(INT32_C( 1425225784), INT32_C( 2110783812)) },

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int32x2_t r = simde_vabs_s32(test_vec[i].a);
    simde_neon_assert_int32x2(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

#if 0
static MunitResult
test_simde_vabs_s64(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int64x1_t a;
    simde_int64x1_t r;
  } test_vec[8] = {

  };

  printf("\n");
  for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) {
    simde_int64x1_t a, r;

    munit_rand_memory(sizeof(a), (uint8_t*) &a);

    r = simde_vabs_s64(a);

    printf("    { simde_x_vload_s64(INT64_C(%20" PRId64 ")),\n", a.i64[0]);
    printf("      simde_x_vload_s64(INT64_C(%20" PRId64 ")) },\n", r.i64[0]);
  }
  return MUNIT_FAIL;

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int64x1_t r = simde_vabs_s64(test_vec[i].a);
    simde_neon_assert_int64x1(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}
#endif


static MunitResult
test_simde_vabs_f32(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_float32x2_t a;
    simde_float32x2_t r;
  } test_vec[8] = {
    { simde_x_vload_f32(SIMDE_FLOAT32_C( -948.69), SIMDE_FLOAT32_C(   59.57)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  948.69), SIMDE_FLOAT32_C(   59.57)) },
    { simde_x_vload_f32(SIMDE_FLOAT32_C(  744.28), SIMDE_FLOAT32_C(  734.52)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  744.28), SIMDE_FLOAT32_C(  734.52)) },
    { simde_x_vload_f32(SIMDE_FLOAT32_C(  -41.62), SIMDE_FLOAT32_C(  162.79)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(   41.62), SIMDE_FLOAT32_C(  162.79)) },
    { simde_x_vload_f32(SIMDE_FLOAT32_C(  396.14), SIMDE_FLOAT32_C(  127.15)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  396.14), SIMDE_FLOAT32_C(  127.15)) },
    { simde_x_vload_f32(SIMDE_FLOAT32_C(  260.62), SIMDE_FLOAT32_C( -846.81)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  260.62), SIMDE_FLOAT32_C(  846.81)) },
    { simde_x_vload_f32(SIMDE_FLOAT32_C(  281.18), SIMDE_FLOAT32_C(  872.09)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  281.18), SIMDE_FLOAT32_C(  872.09)) },
    { simde_x_vload_f32(SIMDE_FLOAT32_C( -306.71), SIMDE_FLOAT32_C(  233.32)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  306.71), SIMDE_FLOAT32_C(  233.32)) },
    { simde_x_vload_f32(SIMDE_FLOAT32_C(  336.33), SIMDE_FLOAT32_C(   17.09)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  336.33), SIMDE_FLOAT32_C(   17.09)) },

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_float32x2_t r = simde_vabs_f32(test_vec[i].a);
    simde_neon_assert_float32x2_equal(r, test_vec[i].r, 1);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabs_f64(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_float64x1_t a;
    simde_float64x1_t r;
  } test_vec[8] = {
    { simde_x_vload_f64(SIMDE_FLOAT64_C( -948.69)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(  948.69)) },
    { simde_x_vload_f64(SIMDE_FLOAT64_C(   59.57)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(   59.57)) },
    { simde_x_vload_f64(SIMDE_FLOAT64_C(  744.28)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(  744.28)) },
    { simde_x_vload_f64(SIMDE_FLOAT64_C(  734.52)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(  734.52)) },
    { simde_x_vload_f64(SIMDE_FLOAT64_C(  -41.62)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(   41.62)) },
    { simde_x_vload_f64(SIMDE_FLOAT64_C(  162.79)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(  162.79)) },
    { simde_x_vload_f64(SIMDE_FLOAT64_C(  396.14)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(  396.14)) },
    { simde_x_vload_f64(SIMDE_FLOAT64_C(  127.15)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(  127.15)) },

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_float64x1_t r = simde_vabs_f64(test_vec[i].a);
    simde_neon_assert_float64x1_equal(r, test_vec[i].r, 1);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabsq_s8(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int8x16_t a;
    simde_int8x16_t r;
  } test_vec[8] = {
    { simde_x_vloadq_s8(INT8_C( 118), INT8_C( 102), INT8_C(-111), INT8_C(   6),
                        INT8_C(  76), INT8_C( -43), INT8_C( -97), INT8_C(-121),
                        INT8_C(   0), INT8_C( -73), INT8_C(  68), INT8_C( -33),
                        INT8_C(  18), INT8_C( -92), INT8_C(   4), INT8_C( -34)),
      simde_x_vloadq_s8(INT8_C( 118), INT8_C( 102), INT8_C( 111), INT8_C(   6),
                        INT8_C(  76), INT8_C(  43), INT8_C(  97), INT8_C( 121),
                        INT8_C(   0), INT8_C(  73), INT8_C(  68), INT8_C(  33),
                        INT8_C(  18), INT8_C(  92), INT8_C(   4), INT8_C(  34)) },
    { simde_x_vloadq_s8(INT8_C( 113), INT8_C(  42), INT8_C( -84), INT8_C( 122),
                        INT8_C(  63), INT8_C(  51), INT8_C( -42), INT8_C(-108),
                        INT8_C(  74), INT8_C( -65), INT8_C( -76), INT8_C( -78),
                        INT8_C(  15), INT8_C( 118), INT8_C(  70), INT8_C(-112)),
      simde_x_vloadq_s8(INT8_C( 113), INT8_C(  42), INT8_C(  84), INT8_C( 122),
                        INT8_C(  63), INT8_C(  51), INT8_C(  42), INT8_C( 108),
                        INT8_C(  74), INT8_C(  65), INT8_C(  76), INT8_C(  78),
                        INT8_C(  15), INT8_C( 118), INT8_C(  70), INT8_C( 112)) },
    { simde_x_vloadq_s8(INT8_C(  28), INT8_C(  -8), INT8_C(  91), INT8_C( -95),
                        INT8_C(  80), INT8_C( -63), INT8_C(-101), INT8_C(  19),
                        INT8_C( -71), INT8_C( -86), INT8_C(  -3), INT8_C( -93),
                        INT8_C( -31), INT8_C(-112), INT8_C( -96), INT8_C( -17)),
      simde_x_vloadq_s8(INT8_C(  28), INT8_C(   8), INT8_C(  91), INT8_C(  95),
                        INT8_C(  80), INT8_C(  63), INT8_C( 101), INT8_C(  19),
                        INT8_C(  71), INT8_C(  86), INT8_C(   3), INT8_C(  93),
                        INT8_C(  31), INT8_C( 112), INT8_C(  96), INT8_C(  17)) },
    { simde_x_vloadq_s8(INT8_C( -45), INT8_C( -96), INT8_C( -67), INT8_C(  88),
                        INT8_C(-126), INT8_C(-122), INT8_C( -35), INT8_C( -99),
                        INT8_C( -56), INT8_C( -57), INT8_C(  12), INT8_C( -85),
                        INT8_C( -68), INT8_C(  -2), INT8_C(  47), INT8_C(-126)),
      simde_x_vloadq_s8(INT8_C(  45), INT8_C(  96), INT8_C(  67), INT8_C(  88),
                        INT8_C( 126), INT8_C( 122), INT8_C(  35), INT8_C(  99),
                        INT8_C(  56), INT8_C(  57), INT8_C(  12), INT8_C(  85),
                        INT8_C(  68), INT8_C(   2), INT8_C(  47), INT8_C( 126)) },
    { simde_x_vloadq_s8(INT8_C( -47), INT8_C(  66), INT8_C(  17), INT8_C(  50),
                        INT8_C(  78), INT8_C(  75), INT8_C(  79), INT8_C(  16),
                        INT8_C(  -4), INT8_C( -96), INT8_C(-113), INT8_C(  93),
                        INT8_C( -36), INT8_C(  13), INT8_C( 119), INT8_C(  57)),
      simde_x_vloadq_s8(INT8_C(  47), INT8_C(  66), INT8_C(  17), INT8_C(  50),
                        INT8_C(  78), INT8_C(  75), INT8_C(  79), INT8_C(  16),
                        INT8_C(   4), INT8_C(  96), INT8_C( 113), INT8_C(  93),
                        INT8_C(  36), INT8_C(  13), INT8_C( 119), INT8_C(  57)) },
    { simde_x_vloadq_s8(INT8_C( -72), INT8_C(  92), INT8_C( -55), INT8_C(-107),
                        INT8_C( -66), INT8_C( -42), INT8_C( -95), INT8_C(-125),
                        INT8_C(  40), INT8_C(  35), INT8_C( -34), INT8_C(  58),
                        INT8_C( 101), INT8_C( -90), INT8_C( -32), INT8_C(-121)),
      simde_x_vloadq_s8(INT8_C(  72), INT8_C(  92), INT8_C(  55), INT8_C( 107),
                        INT8_C(  66), INT8_C(  42), INT8_C(  95), INT8_C( 125),
                        INT8_C(  40), INT8_C(  35), INT8_C(  34), INT8_C(  58),
                        INT8_C( 101), INT8_C(  90), INT8_C(  32), INT8_C( 121)) },
    { simde_x_vloadq_s8(INT8_C( 117), INT8_C(  27), INT8_C( -48), INT8_C( 117),
                        INT8_C(  75), INT8_C( -53), INT8_C(  93), INT8_C(  40),
                        INT8_C( -21), INT8_C( -46), INT8_C(  67), INT8_C( -53),
                        INT8_C( 101), INT8_C(-127), INT8_C( 107), INT8_C(  33)),
      simde_x_vloadq_s8(INT8_C( 117), INT8_C(  27), INT8_C(  48), INT8_C( 117),
                        INT8_C(  75), INT8_C(  53), INT8_C(  93), INT8_C(  40),
                        INT8_C(  21), INT8_C(  46), INT8_C(  67), INT8_C(  53),
                        INT8_C( 101), INT8_C( 127), INT8_C( 107), INT8_C(  33)) },
    { simde_x_vloadq_s8(INT8_C( -62), INT8_C(  79), INT8_C(-105), INT8_C( -99),
                        INT8_C(  80), INT8_C(  -7), INT8_C( -46), INT8_C(  33),
                        INT8_C(-127), INT8_C( -24), INT8_C(-111), INT8_C(  -6),
                        INT8_C( -87), INT8_C(  42), INT8_C(  37), INT8_C( -78)),
      simde_x_vloadq_s8(INT8_C(  62), INT8_C(  79), INT8_C( 105), INT8_C(  99),
                        INT8_C(  80), INT8_C(   7), INT8_C(  46), INT8_C(  33),
                        INT8_C( 127), INT8_C(  24), INT8_C( 111), INT8_C(   6),
                        INT8_C(  87), INT8_C(  42), INT8_C(  37), INT8_C(  78)) },

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int8x16_t r = simde_vabsq_s8(test_vec[i].a);
    simde_neon_assert_int8x16(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabsq_s16(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int16x8_t a;
    simde_int16x8_t r;
  } test_vec[8] = {
    { simde_x_vloadq_s16(INT16_C( 26230), INT16_C(  1681), INT16_C(-10932), INT16_C(-30817),
                         INT16_C(-18688), INT16_C( -8380), INT16_C(-23534), INT16_C( -8700)),
      simde_x_vloadq_s16(INT16_C( 26230), INT16_C(  1681), INT16_C( 10932), INT16_C( 30817),
                         INT16_C( 18688), INT16_C(  8380), INT16_C( 23534), INT16_C(  8700)) },
    { simde_x_vloadq_s16(INT16_C( 10865), INT16_C( 31404), INT16_C( 13119), INT16_C(-27434),
                         INT16_C(-16566), INT16_C(-19788), INT16_C( 30223), INT16_C(-28602)),
      simde_x_vloadq_s16(INT16_C( 10865), INT16_C( 31404), INT16_C( 13119), INT16_C( 27434),
                         INT16_C( 16566), INT16_C( 19788), INT16_C( 30223), INT16_C( 28602)) },
    { simde_x_vloadq_s16(INT16_C( -2020), INT16_C(-24229), INT16_C(-16048), INT16_C(  5019),
                         INT16_C(-21831), INT16_C(-23555), INT16_C(-28447), INT16_C( -4192)),
      simde_x_vloadq_s16(INT16_C(  2020), INT16_C( 24229), INT16_C( 16048), INT16_C(  5019),
                         INT16_C( 21831), INT16_C( 23555), INT16_C( 28447), INT16_C(  4192)) },
    { simde_x_vloadq_s16(INT16_C(-24365), INT16_C( 22717), INT16_C(-31102), INT16_C(-25123),
                         INT16_C(-14392), INT16_C(-21748), INT16_C(  -324), INT16_C(-32209)),
      simde_x_vloadq_s16(INT16_C( 24365), INT16_C( 22717), INT16_C( 31102), INT16_C( 25123),
                         INT16_C( 14392), INT16_C( 21748), INT16_C(   324), INT16_C( 32209)) },
    { simde_x_vloadq_s16(INT16_C( 17105), INT16_C( 12817), INT16_C( 19278), INT16_C(  4175),
                         INT16_C(-24324), INT16_C( 23951), INT16_C(  3548), INT16_C( 14711)),
      simde_x_vloadq_s16(INT16_C( 17105), INT16_C( 12817), INT16_C( 19278), INT16_C(  4175),
                         INT16_C( 24324), INT16_C( 23951), INT16_C(  3548), INT16_C( 14711)) },
    { simde_x_vloadq_s16(INT16_C( 23736), INT16_C(-27191), INT16_C(-10562), INT16_C(-31839),
                         INT16_C(  9000), INT16_C( 15070), INT16_C(-22939), INT16_C(-30752)),
      simde_x_vloadq_s16(INT16_C( 23736), INT16_C( 27191), INT16_C( 10562), INT16_C( 31839),
                         INT16_C(  9000), INT16_C( 15070), INT16_C( 22939), INT16_C( 30752)) },
    { simde_x_vloadq_s16(INT16_C(  7029), INT16_C( 30160), INT16_C(-13493), INT16_C( 10333),
                         INT16_C(-11541), INT16_C(-13501), INT16_C(-32411), INT16_C(  8555)),
      simde_x_vloadq_s16(INT16_C(  7029), INT16_C( 30160), INT16_C( 13493), INT16_C( 10333),
                         INT16_C( 11541), INT16_C( 13501), INT16_C( 32411), INT16_C(  8555)) },
    { simde_x_vloadq_s16(INT16_C( 20418), INT16_C(-25193), INT16_C( -1712), INT16_C(  8658),
                         INT16_C( -6015), INT16_C( -1391), INT16_C( 10921), INT16_C(-19931)),
      simde_x_vloadq_s16(INT16_C( 20418), INT16_C( 25193), INT16_C(  1712), INT16_C(  8658),
                         INT16_C(  6015), INT16_C(  1391), INT16_C( 10921), INT16_C( 19931)) },

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int16x8_t r = simde_vabsq_s16(test_vec[i].a);
    simde_neon_assert_int16x8(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabsq_s32(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int32x4_t a;
    simde_int32x4_t r;
  } test_vec[8] = {
    { simde_x_vloadq_s32(INT32_C(  110192246), INT32_C(-2019568308),
                         INT32_C( -549144832), INT32_C( -570121198)),
      simde_x_vloadq_s32(INT32_C(  110192246), INT32_C( 2019568308),
                         INT32_C(  549144832), INT32_C(  570121198)) },
    { simde_x_vloadq_s32(INT32_C( 2058103409), INT32_C(-1797901505),
                         INT32_C(-1296777398), INT32_C(-1874430449)),
      simde_x_vloadq_s32(INT32_C( 2058103409), INT32_C( 1797901505),
                         INT32_C( 1296777398), INT32_C( 1874430449)) },
    { simde_x_vloadq_s32(INT32_C(-1587808228), INT32_C(  328974672),
                         INT32_C(-1543656775), INT32_C( -274689823)),
      simde_x_vloadq_s32(INT32_C( 1587808228), INT32_C(  328974672),
                         INT32_C( 1543656775), INT32_C(  274689823)) },
    { simde_x_vloadq_s32(INT32_C( 1488822483), INT32_C(-1646426494),
                         INT32_C(-1425225784), INT32_C(-2110783812)),
      simde_x_vloadq_s32(INT32_C( 1488822483), INT32_C( 1646426494),
                         INT32_C( 1425225784), INT32_C( 2110783812)) },
    { simde_x_vloadq_s32(INT32_C(  839992017), INT32_C(  273632078),
                         INT32_C( 1569693948), INT32_C(  964103644)),
      simde_x_vloadq_s32(INT32_C(  839992017), INT32_C(  273632078),
                         INT32_C( 1569693948), INT32_C(  964103644)) },
    { simde_x_vloadq_s32(INT32_C(-1781965640), INT32_C(-2086545730),
                         INT32_C(  987636520), INT32_C(-2015320475)),
      simde_x_vloadq_s32(INT32_C( 1781965640), INT32_C( 2086545730),
                         INT32_C(  987636520), INT32_C( 2015320475)) },
    { simde_x_vloadq_s32(INT32_C( 1976572789), INT32_C(  677235531),
                         INT32_C( -884747541), INT32_C(  560693605)),
      simde_x_vloadq_s32(INT32_C( 1976572789), INT32_C(  677235531),
                         INT32_C(  884747541), INT32_C(  560693605)) },
    { simde_x_vloadq_s32(INT32_C(-1651028030), INT32_C(  567474512),
                         INT32_C(  -91101055), INT32_C(-1306187095)),
      simde_x_vloadq_s32(INT32_C( 1651028030), INT32_C(  567474512),
                         INT32_C(   91101055), INT32_C( 1306187095)) },

  };





  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int32x4_t r = simde_vabsq_s32(test_vec[i].a);
    simde_neon_assert_int32x4(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

#if 0
static MunitResult
test_simde_vabsq_s64(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int64x2_t a;
    simde_int64x2_t b;
    simde_int64x2_t r;
  } test_vec[8] = {

  };

  printf("\n");
  for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) {
    simde_int64x2_t a, r;

    munit_rand_memory(sizeof(a), (uint8_t*) &a);

    r = simde_vabsq_s64(a);

    printf("    { simde_x_vloadq_s64(INT64_C(%21" PRId64 "), INT64_C(%21" PRId64 ")),\n", a.i64[0], a.i64[1]);
    printf("      simde_x_vloadq_s64(INT64_C(%21" PRId64 "), INT64_C(%21" PRId64 ")) },\n", r.i64[0], r.i64[1]);
  }
  return MUNIT_FAIL;

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int64x2_t r = simde_vabsq_s64(test_vec[i].a);
    simde_neon_assert_int64x2(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}
#endif

static MunitResult
test_simde_vabsq_f32(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_float32x4_t a;
    simde_float32x4_t r;
  } test_vec[8] = {
    { simde_x_vloadq_f32(SIMDE_FLOAT32_C( -948.69), SIMDE_FLOAT32_C(   59.57),
                         SIMDE_FLOAT32_C(  744.28), SIMDE_FLOAT32_C(  734.52)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  948.69), SIMDE_FLOAT32_C(   59.57),
                         SIMDE_FLOAT32_C(  744.28), SIMDE_FLOAT32_C(  734.52)) },
    { simde_x_vloadq_f32(SIMDE_FLOAT32_C(  -41.62), SIMDE_FLOAT32_C(  162.79),
                         SIMDE_FLOAT32_C(  396.14), SIMDE_FLOAT32_C(  127.15)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(   41.62), SIMDE_FLOAT32_C(  162.79),
                         SIMDE_FLOAT32_C(  396.14), SIMDE_FLOAT32_C(  127.15)) },
    { simde_x_vloadq_f32(SIMDE_FLOAT32_C(  260.62), SIMDE_FLOAT32_C( -846.81),
                         SIMDE_FLOAT32_C(  281.18), SIMDE_FLOAT32_C(  872.09)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  260.62), SIMDE_FLOAT32_C(  846.81),
                         SIMDE_FLOAT32_C(  281.18), SIMDE_FLOAT32_C(  872.09)) },
    { simde_x_vloadq_f32(SIMDE_FLOAT32_C( -306.71), SIMDE_FLOAT32_C(  233.32),
                         SIMDE_FLOAT32_C(  336.33), SIMDE_FLOAT32_C(   17.09)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  306.71), SIMDE_FLOAT32_C(  233.32),
                         SIMDE_FLOAT32_C(  336.33), SIMDE_FLOAT32_C(   17.09)) },
    { simde_x_vloadq_f32(SIMDE_FLOAT32_C( -608.85), SIMDE_FLOAT32_C( -872.58),
                         SIMDE_FLOAT32_C( -269.05), SIMDE_FLOAT32_C( -551.05)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  608.85), SIMDE_FLOAT32_C(  872.58),
                         SIMDE_FLOAT32_C(  269.05), SIMDE_FLOAT32_C(  551.05)) },
    { simde_x_vloadq_f32(SIMDE_FLOAT32_C(  170.21), SIMDE_FLOAT32_C(   28.38),
                         SIMDE_FLOAT32_C( -540.10), SIMDE_FLOAT32_C(   61.54)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  170.21), SIMDE_FLOAT32_C(   28.38),
                         SIMDE_FLOAT32_C(  540.10), SIMDE_FLOAT32_C(   61.54)) },
    { simde_x_vloadq_f32(SIMDE_FLOAT32_C(  -79.59), SIMDE_FLOAT32_C( -684.64),
                         SIMDE_FLOAT32_C(  588.01), SIMDE_FLOAT32_C( -738.91)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(   79.59), SIMDE_FLOAT32_C(  684.64),
                         SIMDE_FLOAT32_C(  588.01), SIMDE_FLOAT32_C(  738.91)) },
    { simde_x_vloadq_f32(SIMDE_FLOAT32_C(  231.18), SIMDE_FLOAT32_C( -735.75),
                         SIMDE_FLOAT32_C(  957.58), SIMDE_FLOAT32_C(  391.76)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  231.18), SIMDE_FLOAT32_C(  735.75),
                         SIMDE_FLOAT32_C(  957.58), SIMDE_FLOAT32_C(  391.76)) },

  };


  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_float32x4_t r = simde_vabsq_f32(test_vec[i].a);
    simde_neon_assert_float32x4_equal(r, test_vec[i].r, 1);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabsq_f64(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_float64x2_t a;
    simde_float64x2_t r;
  } test_vec[8] = {
    { simde_x_vloadq_f64(SIMDE_FLOAT64_C( -948.69), SIMDE_FLOAT64_C(   59.57)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  948.69), SIMDE_FLOAT64_C(   59.57)) },
    { simde_x_vloadq_f64(SIMDE_FLOAT64_C(  744.28), SIMDE_FLOAT64_C(  734.52)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  744.28), SIMDE_FLOAT64_C(  734.52)) },
    { simde_x_vloadq_f64(SIMDE_FLOAT64_C(  -41.62), SIMDE_FLOAT64_C(  162.79)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(   41.62), SIMDE_FLOAT64_C(  162.79)) },
    { simde_x_vloadq_f64(SIMDE_FLOAT64_C(  396.14), SIMDE_FLOAT64_C(  127.15)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  396.14), SIMDE_FLOAT64_C(  127.15)) },
    { simde_x_vloadq_f64(SIMDE_FLOAT64_C(  260.62), SIMDE_FLOAT64_C( -846.81)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  260.62), SIMDE_FLOAT64_C(  846.81)) },
    { simde_x_vloadq_f64(SIMDE_FLOAT64_C(  281.18), SIMDE_FLOAT64_C(  872.09)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  281.18), SIMDE_FLOAT64_C(  872.09)) },
    { simde_x_vloadq_f64(SIMDE_FLOAT64_C( -306.71), SIMDE_FLOAT64_C(  233.32)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  306.71), SIMDE_FLOAT64_C(  233.32)) },
    { simde_x_vloadq_f64(SIMDE_FLOAT64_C(  336.33), SIMDE_FLOAT64_C(   17.09)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  336.33), SIMDE_FLOAT64_C(   17.09)) },

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_float64x2_t r = simde_vabsq_f64(test_vec[i].a);
    simde_neon_assert_float64x2_equal(r, test_vec[i].r, 1);
  }

  return MUNIT_OK;
}

HEDLEY_DIAGNOSTIC_PUSH
HEDLEY_DIAGNOSTIC_DISABLE_CAST_QUAL

static MunitTest abs_tests[] = {
  SIMDE_TESTS_NEON_DEFINE_TEST(s8),
  SIMDE_TESTS_NEON_DEFINE_TEST(s16),
  SIMDE_TESTS_NEON_DEFINE_TEST(s32),
  /* SIMDE_TESTS_NEON_DEFINE_TEST(s64), */
  /* SIMDE_TESTS_NEON_DEFINE_TEST(u64), */
  SIMDE_TESTS_NEON_DEFINE_TEST(f32),
  SIMDE_TESTS_NEON_DEFINE_TEST(f64),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, s8),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, s16),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, s32),
  /* SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, s64), */
  /* SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, u64), */
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, f32),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, f64),

  { NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL }
};

HEDLEY_C_DECL MunitSuite* SIMDE_TESTS_GENERATE_SYMBOL(SIMDE_TESTS_CURRENT_NEON_OP)(void) {
  static MunitSuite suite = { (char*) "/v" HEDLEY_STRINGIFY(SIMDE_TESTS_CURRENT_NEON_OP), abs_tests, NULL, 1, MUNIT_SUITE_OPTION_NONE };

  return &suite;
}

HEDLEY_DIAGNOSTIC_POP
