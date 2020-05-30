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

#define SIMDE_TESTS_CURRENT_NEON_OP aba
#include <test/arm/neon/test-neon-internal.h>
#include <simde/arm/neon.h>

static MunitResult
test_simde_vaba_s8(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int8x8_t a;
    simde_int8x8_t b;
    simde_int8x8_t c;
    simde_int8x8_t r;
  } test_vec[8] = {
    { simde_x_vload_s8(INT8_C(  38), INT8_C( -47), INT8_C( -85), INT8_C( -33),
                       INT8_C(  67), INT8_C(  87), INT8_C( -85), INT8_C(  49)),
      simde_x_vload_s8(INT8_C( 120), INT8_C( -89), INT8_C( -42), INT8_C(-111),
                       INT8_C( -89), INT8_C( -46), INT8_C(  79), INT8_C( -82)),
      simde_x_vload_s8(INT8_C(  65), INT8_C(-111), INT8_C( -49), INT8_C( 110),
                       INT8_C(  44), INT8_C( -53), INT8_C( 102), INT8_C(  30)),
      simde_x_vload_s8(INT8_C(  93), INT8_C( -25), INT8_C( -78), INT8_C( -68),
                       INT8_C( -56), INT8_C(  94), INT8_C( -62), INT8_C( -95)) },
    { simde_x_vload_s8(INT8_C( -55), INT8_C(  86), INT8_C( -63), INT8_C(  15),
                       INT8_C( -89), INT8_C(  52), INT8_C(  77), INT8_C(  67)),
      simde_x_vload_s8(INT8_C(  38), INT8_C( -46), INT8_C( -41), INT8_C(  39),
                       INT8_C( -69), INT8_C(  26), INT8_C(-114), INT8_C(   0)),
      simde_x_vload_s8(INT8_C(  35), INT8_C(  99), INT8_C( -99), INT8_C(  -4),
                       INT8_C(   1), INT8_C(  93), INT8_C(  40), INT8_C( -94)),
      simde_x_vload_s8(INT8_C( -52), INT8_C( -25), INT8_C(  -5), INT8_C(  58),
                       INT8_C( -19), INT8_C( 119), INT8_C( -25), INT8_C( -95)) },
    { simde_x_vload_s8(INT8_C( -46), INT8_C(  18), INT8_C( 103), INT8_C( -85),
                       INT8_C(-121), INT8_C( -46), INT8_C( 116), INT8_C(-120)),
      simde_x_vload_s8(INT8_C(  18), INT8_C( -28), INT8_C(   3), INT8_C( -53),
                       INT8_C( -18), INT8_C( -74), INT8_C(  43), INT8_C( -83)),
      simde_x_vload_s8(INT8_C(  -5), INT8_C(  68), INT8_C( 100), INT8_C(  40),
                       INT8_C(  58), INT8_C(  25), INT8_C(-120), INT8_C(-124)),
      simde_x_vload_s8(INT8_C( -23), INT8_C( 114), INT8_C( -56), INT8_C(   8),
                       INT8_C( -45), INT8_C(  53), INT8_C(  23), INT8_C( -79)) },
    { simde_x_vload_s8(INT8_C(  52), INT8_C( -87), INT8_C( -81), INT8_C( -82),
                       INT8_C( 111), INT8_C( -54), INT8_C(  69), INT8_C( -87)),
      simde_x_vload_s8(INT8_C(-114), INT8_C( 123), INT8_C(  80), INT8_C( -39),
                       INT8_C( -29), INT8_C(-101), INT8_C( -40), INT8_C( -60)),
      simde_x_vload_s8(INT8_C(  21), INT8_C(  24), INT8_C(  -2), INT8_C(  93),
                       INT8_C(  46), INT8_C(  16), INT8_C(  92), INT8_C(  23)),
      simde_x_vload_s8(INT8_C( -69), INT8_C(  12), INT8_C(   1), INT8_C(  50),
                       INT8_C( -70), INT8_C(  63), INT8_C( -55), INT8_C(  -4)) },
    { simde_x_vload_s8(INT8_C( -97), INT8_C(  96), INT8_C(  96), INT8_C(   9),
                       INT8_C(   5), INT8_C( 108), INT8_C(-103), INT8_C(  66)),
      simde_x_vload_s8(INT8_C( -21), INT8_C(  86), INT8_C( 115), INT8_C( -44),
                       INT8_C( 104), INT8_C( -26), INT8_C(  50), INT8_C( -92)),
      simde_x_vload_s8(INT8_C(  43), INT8_C( -38), INT8_C(  54), INT8_C(   0),
                       INT8_C(  22), INT8_C(  23), INT8_C(  92), INT8_C(  -8)),
      simde_x_vload_s8(INT8_C( -33), INT8_C( -36), INT8_C( -99), INT8_C(  53),
                       INT8_C(  87), INT8_C( -99), INT8_C( -61), INT8_C(-106)) },
    { simde_x_vload_s8(INT8_C(  69), INT8_C(  60), INT8_C(  84), INT8_C( -59),
                       INT8_C( -20), INT8_C( 120), INT8_C( -42), INT8_C(  26)),
      simde_x_vload_s8(INT8_C( -44), INT8_C(  50), INT8_C( -98), INT8_C(  14),
                       INT8_C(  24), INT8_C( -44), INT8_C( -51), INT8_C(-102)),
      simde_x_vload_s8(INT8_C(  58), INT8_C(  53), INT8_C(  85), INT8_C(  25),
                       INT8_C(   1), INT8_C(  98), INT8_C( -97), INT8_C( -70)),
      simde_x_vload_s8(INT8_C( -85), INT8_C(  63), INT8_C(  11), INT8_C( -48),
                       INT8_C(   3), INT8_C(   6), INT8_C(   4), INT8_C(  58)) },
    { simde_x_vload_s8(INT8_C( -33), INT8_C(-108), INT8_C( 100), INT8_C( -11),
                       INT8_C(  46), INT8_C( -83), INT8_C(-115), INT8_C(  26)),
      simde_x_vload_s8(INT8_C( 123), INT8_C(   9), INT8_C(  60), INT8_C(  23),
                       INT8_C(  52), INT8_C( -83), INT8_C(  53), INT8_C(  10)),
      simde_x_vload_s8(INT8_C( -63), INT8_C(  21), INT8_C(   8), INT8_C( 105),
                       INT8_C( -56), INT8_C(   6), INT8_C(  36), INT8_C(  22)),
      simde_x_vload_s8(INT8_C(-103), INT8_C( -96), INT8_C(-104), INT8_C(  71),
                       INT8_C(-102), INT8_C(   6), INT8_C( -98), INT8_C(  38)) },
    { simde_x_vload_s8(INT8_C( -70), INT8_C( 122), INT8_C( -65), INT8_C( -48),
                       INT8_C( -81), INT8_C(-121), INT8_C( -79), INT8_C(  66)),
      simde_x_vload_s8(INT8_C( -98), INT8_C(-120), INT8_C( -90), INT8_C( -16),
                       INT8_C(  32), INT8_C( -75), INT8_C( 101), INT8_C( -37)),
      simde_x_vload_s8(INT8_C(-104), INT8_C( 109), INT8_C( 116), INT8_C( 105),
                       INT8_C( -12), INT8_C(-117), INT8_C( -82), INT8_C(  64)),
      simde_x_vload_s8(INT8_C( -64), INT8_C(  95), INT8_C(-115), INT8_C(  73),
                       INT8_C( -37), INT8_C( -79), INT8_C( 104), INT8_C( -89)) },

  };

  /* printf("\n"); */
  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_int8x8_t a, b, c, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */
  /*   munit_rand_memory(sizeof(c), (uint8_t*) &c); */

  /*   r = simde_vaba_s8(a, b, c); */

  /*   printf("    { simde_x_vload_s8(INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                       INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 ")),\n", */
  /*          a.i8[0], a.i8[1], a.i8[2], a.i8[3], a.i8[4], a.i8[5], a.i8[6], a.i8[7]); */
  /*   printf("      simde_x_vload_s8(INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                       INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 ")),\n", */
  /*          b.i8[0], b.i8[1], b.i8[2], b.i8[3], b.i8[4], b.i8[5], b.i8[6], b.i8[7]); */
  /*   printf("      simde_x_vload_s8(INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                       INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 ")),\n", */
  /*          c.i8[0], c.i8[1], c.i8[2], c.i8[3], c.i8[4], c.i8[5], c.i8[6], c.i8[7]); */
  /*   printf("      simde_x_vload_s8(INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                       INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 ")) },\n", */
  /*          r.i8[0], r.i8[1], r.i8[2], r.i8[3], r.i8[4], r.i8[5], r.i8[6], r.i8[7]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int8x8_t r = simde_vaba_s8(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    simde_neon_assert_int8x8(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vaba_s16(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int16x4_t a;
    simde_int16x4_t b;
    simde_int16x4_t c;
    simde_int16x4_t r;
  } test_vec[8] = {
    { simde_x_vload_s16(INT16_C(-11994), INT16_C( -8277), INT16_C( 22339), INT16_C( 12715)),
      simde_x_vload_s16(INT16_C(-22664), INT16_C(-28202), INT16_C(-11609), INT16_C(-20913)),
      simde_x_vload_s16(INT16_C(-28351), INT16_C( 28367), INT16_C(-13524), INT16_C(  7782)),
      simde_x_vload_s16(INT16_C( -6307), INT16_C(-17244), INT16_C( 24254), INT16_C(-24126)) },
    { simde_x_vload_s16(INT16_C( 22217), INT16_C(  4033), INT16_C( 13479), INT16_C( 17229)),
      simde_x_vload_s16(INT16_C(-11738), INT16_C( 10199), INT16_C(  6843), INT16_C(   142)),
      simde_x_vload_s16(INT16_C( 25379), INT16_C(  -867), INT16_C( 23809), INT16_C(-24024)),
      simde_x_vload_s16(INT16_C( -6202), INT16_C( 15099), INT16_C( 30445), INT16_C(-24141)) },
    { simde_x_vload_s16(INT16_C(  4818), INT16_C(-21657), INT16_C(-11641), INT16_C(-30604)),
      simde_x_vload_s16(INT16_C( -7150), INT16_C(-13565), INT16_C(-18706), INT16_C(-21205)),
      simde_x_vload_s16(INT16_C( 17659), INT16_C( 10340), INT16_C(  6458), INT16_C(-31608)),
      simde_x_vload_s16(INT16_C( 29627), INT16_C(  2248), INT16_C( 13523), INT16_C(-20201)) },
    { simde_x_vload_s16(INT16_C(-22220), INT16_C(-20817), INT16_C(-13713), INT16_C(-22203)),
      simde_x_vload_s16(INT16_C( 31630), INT16_C( -9904), INT16_C(-25629), INT16_C(-15144)),
      simde_x_vload_s16(INT16_C(  6165), INT16_C( 24062), INT16_C(  4142), INT16_C(  5980)),
      simde_x_vload_s16(INT16_C(  3245), INT16_C( 13149), INT16_C( 16058), INT16_C( -1079)) },
    { simde_x_vload_s16(INT16_C( 24735), INT16_C(  2400), INT16_C( 27653), INT16_C( 17049)),
      simde_x_vload_s16(INT16_C( 22251), INT16_C(-11149), INT16_C( -6552), INT16_C(-23502)),
      simde_x_vload_s16(INT16_C( -9685), INT16_C(    54), INT16_C(  5910), INT16_C( -1956)),
      simde_x_vload_s16(INT16_C( -8865), INT16_C( 13603), INT16_C(-25421), INT16_C(-26941)) },
    { simde_x_vload_s16(INT16_C( 15429), INT16_C(-15020), INT16_C( 30956), INT16_C(  6870)),
      simde_x_vload_s16(INT16_C( 13012), INT16_C(  3742), INT16_C(-11240), INT16_C(-25907)),
      simde_x_vload_s16(INT16_C( 13626), INT16_C(  6485), INT16_C( 25089), INT16_C(-17761)),
      simde_x_vload_s16(INT16_C( 16043), INT16_C(-12277), INT16_C(  1749), INT16_C( 15016)) },
    { simde_x_vload_s16(INT16_C(-27425), INT16_C( -2716), INT16_C(-21202), INT16_C(  6797)),
      simde_x_vload_s16(INT16_C(  2427), INT16_C(  5948), INT16_C(-21196), INT16_C(  2613)),
      simde_x_vload_s16(INT16_C(  5569), INT16_C( 26888), INT16_C(  1736), INT16_C(  5668)),
      simde_x_vload_s16(INT16_C(-24283), INT16_C( 18224), INT16_C(  1730), INT16_C(  9852)) },
    { simde_x_vload_s16(INT16_C( 31418), INT16_C(-12097), INT16_C(-30801), INT16_C( 17073)),
      simde_x_vload_s16(INT16_C(-30562), INT16_C( -3930), INT16_C(-19168), INT16_C( -9371)),
      simde_x_vload_s16(INT16_C( 28056), INT16_C( 26996), INT16_C(-29708), INT16_C( 16558)),
      simde_x_vload_s16(INT16_C( 24500), INT16_C( 18829), INT16_C(-20261), INT16_C(-22534)) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_int16x4_t a, b, c, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */
  /*   munit_rand_memory(sizeof(c), (uint8_t*) &c); */

  /*   r = simde_vaba_s16(a, b, c); */

  /*   printf("    { simde_x_vload_s16(INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 ")),\n", */
  /*          a.i16[0], a.i16[1], a.i16[2], a.i16[3]); */
  /*   printf("      simde_x_vload_s16(INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 ")),\n", */
  /*          b.i16[0], b.i16[1], b.i16[2], b.i16[3]); */
  /*   printf("      simde_x_vload_s16(INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 ")),\n", */
  /*          c.i16[0], c.i16[1], c.i16[2], c.i16[3]); */
  /*   printf("      simde_x_vload_s16(INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 ")) },\n", */
  /*          r.i16[0], r.i16[1], r.i16[2], r.i16[3]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int16x4_t r = simde_vaba_s16(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    simde_neon_assert_int16x4(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vaba_s32(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int32x2_t a;
    simde_int32x2_t b;
    simde_int32x2_t c;
    simde_int32x2_t r;
  } test_vec[8] = {
    { simde_x_vload_s32(INT32_C( -542387930), INT32_C(  833312579)),
      simde_x_vload_s32(INT32_C(-1848203400), INT32_C(-1370500441)),
      simde_x_vload_s32(INT32_C( 1859096897), INT32_C(  510053164)),
      simde_x_vload_s32(INT32_C(   45279069), INT32_C(-1581101112)) },
    { simde_x_vload_s32(INT32_C(  264328905), INT32_C( 1129133223)),
      simde_x_vload_s32(INT32_C(  668455462), INT32_C(    9312955)),
      simde_x_vload_s32(INT32_C(  -56794333), INT32_C(-1574413055)),
      simde_x_vload_s32(INT32_C(  989578700), INT32_C(-1582108063)) },
    { simde_x_vload_s32(INT32_C(-1419308334), INT32_C(-2005609849)),
      simde_x_vload_s32(INT32_C( -888937454), INT32_C(-1389644050)),
      simde_x_vload_s32(INT32_C(  677659899), INT32_C(-2071455430)),
      simde_x_vload_s32(INT32_C(  147289019), INT32_C(-1323798469)) },
    { simde_x_vload_s32(INT32_C(-1364219596), INT32_C(-1455043985)),
      simde_x_vload_s32(INT32_C( -649036914), INT32_C( -992437277)),
      simde_x_vload_s32(INT32_C( 1576933397), INT32_C(  391909422)),
      simde_x_vload_s32(INT32_C(  704777389), INT32_C(  -70697286)) },
    { simde_x_vload_s32(INT32_C(  157311135), INT32_C( 1117350917)),
      simde_x_vload_s32(INT32_C( -730638613), INT32_C(-1540168088)),
      simde_x_vload_s32(INT32_C(    3594795), INT32_C( -128182506)),
      simde_x_vload_s32(INT32_C(  891544543), INT32_C(-1765630797)) },
    { simde_x_vload_s32(INT32_C( -984335291), INT32_C(  450263276)),
      simde_x_vload_s32(INT32_C(  245248724), INT32_C(-1697786856)),
      simde_x_vload_s32(INT32_C(  425014586), INT32_C(-1163959807)),
      simde_x_vload_s32(INT32_C( -804569429), INT32_C(  984090325)) },
    { simde_x_vload_s32(INT32_C( -177957665), INT32_C(  445492526)),
      simde_x_vload_s32(INT32_C(  389810555), INT32_C(  171289908)),
      simde_x_vload_s32(INT32_C( 1762137537), INT32_C(  371459784)),
      simde_x_vload_s32(INT32_C( 1194369317), INT32_C(  645662402)) },
    { simde_x_vload_s32(INT32_C( -792757574), INT32_C( 1118930863)),
      simde_x_vload_s32(INT32_C( -257521506), INT32_C( -614091488)),
      simde_x_vload_s32(INT32_C( 1769237912), INT32_C( 1085180916)),
      simde_x_vload_s32(INT32_C( 1234001844), INT32_C(-1476764029)) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_int32x2_t a, b, c, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */
  /*   munit_rand_memory(sizeof(c), (uint8_t*) &c); */

  /*   r = simde_vaba_s32(a, b, c); */

  /*   printf("    { simde_x_vload_s32(INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 ")),\n", */
  /*          a.i32[0], a.i32[1]); */
  /*   printf("      simde_x_vload_s32(INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 ")),\n", */
  /*          b.i32[0], b.i32[1]); */
  /*   printf("      simde_x_vload_s32(INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 ")),\n", */
  /*          c.i32[0], c.i32[1]); */
  /*   printf("      simde_x_vload_s32(INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 ")) },\n", */
  /*          r.i32[0], r.i32[1]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int32x2_t r = simde_vaba_s32(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    simde_neon_assert_int32x2(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vaba_u8(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_uint8x8_t a;
    simde_uint8x8_t b;
    simde_uint8x8_t c;
    simde_uint8x8_t r;
  } test_vec[8] = {
    { simde_x_vload_u8(UINT8_C( 38), UINT8_C(209), UINT8_C(171), UINT8_C(223),
                       UINT8_C( 67), UINT8_C( 87), UINT8_C(171), UINT8_C( 49)),
      simde_x_vload_u8(UINT8_C(120), UINT8_C(167), UINT8_C(214), UINT8_C(145),
                       UINT8_C(167), UINT8_C(210), UINT8_C( 79), UINT8_C(174)),
      simde_x_vload_u8(UINT8_C( 65), UINT8_C(145), UINT8_C(207), UINT8_C(110),
                       UINT8_C( 44), UINT8_C(203), UINT8_C(102), UINT8_C( 30)),
      simde_x_vload_u8(UINT8_C(239), UINT8_C(186), UINT8_C(164), UINT8_C(188),
                       UINT8_C(200), UINT8_C( 79), UINT8_C(194), UINT8_C(161)) },
    { simde_x_vload_u8(UINT8_C(201), UINT8_C( 86), UINT8_C(193), UINT8_C( 15),
                       UINT8_C(167), UINT8_C( 52), UINT8_C( 77), UINT8_C( 67)),
      simde_x_vload_u8(UINT8_C( 38), UINT8_C(210), UINT8_C(215), UINT8_C( 39),
                       UINT8_C(187), UINT8_C( 26), UINT8_C(142), UINT8_C(  0)),
      simde_x_vload_u8(UINT8_C( 35), UINT8_C( 99), UINT8_C(157), UINT8_C(252),
                       UINT8_C(  1), UINT8_C( 93), UINT8_C( 40), UINT8_C(162)),
      simde_x_vload_u8(UINT8_C(198), UINT8_C(231), UINT8_C(135), UINT8_C(228),
                       UINT8_C(237), UINT8_C(118), UINT8_C(231), UINT8_C(228)) },
    { simde_x_vload_u8(UINT8_C(210), UINT8_C( 18), UINT8_C(103), UINT8_C(171),
                       UINT8_C(135), UINT8_C(210), UINT8_C(116), UINT8_C(136)),
      simde_x_vload_u8(UINT8_C( 18), UINT8_C(228), UINT8_C(  3), UINT8_C(203),
                       UINT8_C(238), UINT8_C(182), UINT8_C( 43), UINT8_C(173)),
      simde_x_vload_u8(UINT8_C(251), UINT8_C( 68), UINT8_C(100), UINT8_C( 40),
                       UINT8_C( 58), UINT8_C( 25), UINT8_C(136), UINT8_C(132)),
      simde_x_vload_u8(UINT8_C(187), UINT8_C(115), UINT8_C(200), UINT8_C(  8),
                       UINT8_C(211), UINT8_C( 52), UINT8_C(209), UINT8_C( 95)) },
    { simde_x_vload_u8(UINT8_C( 52), UINT8_C(169), UINT8_C(175), UINT8_C(174),
                       UINT8_C(111), UINT8_C(202), UINT8_C( 69), UINT8_C(169)),
      simde_x_vload_u8(UINT8_C(142), UINT8_C(123), UINT8_C( 80), UINT8_C(217),
                       UINT8_C(227), UINT8_C(155), UINT8_C(216), UINT8_C(196)),
      simde_x_vload_u8(UINT8_C( 21), UINT8_C( 24), UINT8_C(254), UINT8_C( 93),
                       UINT8_C( 46), UINT8_C( 16), UINT8_C( 92), UINT8_C( 23)),
      simde_x_vload_u8(UINT8_C(187), UINT8_C( 69), UINT8_C( 93), UINT8_C( 51),
                       UINT8_C(186), UINT8_C( 62), UINT8_C(201), UINT8_C(251)) },
    { simde_x_vload_u8(UINT8_C(159), UINT8_C( 96), UINT8_C( 96), UINT8_C(  9),
                       UINT8_C(  5), UINT8_C(108), UINT8_C(153), UINT8_C( 66)),
      simde_x_vload_u8(UINT8_C(235), UINT8_C( 86), UINT8_C(115), UINT8_C(212),
                       UINT8_C(104), UINT8_C(230), UINT8_C( 50), UINT8_C(164)),
      simde_x_vload_u8(UINT8_C( 43), UINT8_C(218), UINT8_C( 54), UINT8_C(  0),
                       UINT8_C( 22), UINT8_C( 23), UINT8_C( 92), UINT8_C(248)),
      simde_x_vload_u8(UINT8_C(223), UINT8_C(227), UINT8_C( 35), UINT8_C( 53),
                       UINT8_C(179), UINT8_C(156), UINT8_C(195), UINT8_C(150)) },
    { simde_x_vload_u8(UINT8_C( 69), UINT8_C( 60), UINT8_C( 84), UINT8_C(197),
                       UINT8_C(236), UINT8_C(120), UINT8_C(214), UINT8_C( 26)),
      simde_x_vload_u8(UINT8_C(212), UINT8_C( 50), UINT8_C(158), UINT8_C( 14),
                       UINT8_C( 24), UINT8_C(212), UINT8_C(205), UINT8_C(154)),
      simde_x_vload_u8(UINT8_C( 58), UINT8_C( 53), UINT8_C( 85), UINT8_C( 25),
                       UINT8_C(  1), UINT8_C( 98), UINT8_C(159), UINT8_C(186)),
      simde_x_vload_u8(UINT8_C(171), UINT8_C( 62), UINT8_C( 11), UINT8_C(208),
                       UINT8_C(213), UINT8_C(  6), UINT8_C(168), UINT8_C( 58)) },
    { simde_x_vload_u8(UINT8_C(223), UINT8_C(148), UINT8_C(100), UINT8_C(245),
                       UINT8_C( 46), UINT8_C(173), UINT8_C(141), UINT8_C( 26)),
      simde_x_vload_u8(UINT8_C(123), UINT8_C(  9), UINT8_C( 60), UINT8_C( 23),
                       UINT8_C( 52), UINT8_C(173), UINT8_C( 53), UINT8_C( 10)),
      simde_x_vload_u8(UINT8_C(193), UINT8_C( 21), UINT8_C(  8), UINT8_C(105),
                       UINT8_C(200), UINT8_C(  6), UINT8_C( 36), UINT8_C( 22)),
      simde_x_vload_u8(UINT8_C( 37), UINT8_C(161), UINT8_C( 48), UINT8_C( 71),
                       UINT8_C(194), UINT8_C(  6), UINT8_C(124), UINT8_C( 38)) },
    { simde_x_vload_u8(UINT8_C(186), UINT8_C(122), UINT8_C(191), UINT8_C(208),
                       UINT8_C(175), UINT8_C(135), UINT8_C(177), UINT8_C( 66)),
      simde_x_vload_u8(UINT8_C(158), UINT8_C(136), UINT8_C(166), UINT8_C(240),
                       UINT8_C( 32), UINT8_C(181), UINT8_C(101), UINT8_C(219)),
      simde_x_vload_u8(UINT8_C(152), UINT8_C(109), UINT8_C(116), UINT8_C(105),
                       UINT8_C(244), UINT8_C(139), UINT8_C(174), UINT8_C( 64)),
      simde_x_vload_u8(UINT8_C(180), UINT8_C( 95), UINT8_C(141), UINT8_C( 73),
                       UINT8_C(131), UINT8_C( 94), UINT8_C(250), UINT8_C(167)) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_uint8x8_t a, b, c, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */
  /*   munit_rand_memory(sizeof(c), (uint8_t*) &c); */

  /*   r = simde_vaba_u8(a, b, c); */

  /*   printf("    { simde_x_vload_u8(UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                       UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 ")),\n", */
  /*          a.u8[0], a.u8[1], a.u8[2], a.u8[3], a.u8[4], a.u8[5], a.u8[6], a.u8[7]); */
  /*   printf("      simde_x_vload_u8(UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                       UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 ")),\n", */
  /*          b.u8[0], b.u8[1], b.u8[2], b.u8[3], b.u8[4], b.u8[5], b.u8[6], b.u8[7]); */
  /*   printf("      simde_x_vload_u8(UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                       UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 ")),\n", */
  /*          c.u8[0], c.u8[1], c.u8[2], c.u8[3], c.u8[4], c.u8[5], c.u8[6], c.u8[7]); */
  /*   printf("      simde_x_vload_u8(UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                       UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 ")) },\n", */
  /*          r.u8[0], r.u8[1], r.u8[2], r.u8[3], r.u8[4], r.u8[5], r.u8[6], r.u8[7]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint8x8_t r = simde_vaba_u8(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    simde_neon_assert_uint8x8(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vaba_u16(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_uint16x4_t a;
    simde_uint16x4_t b;
    simde_uint16x4_t c;
    simde_uint16x4_t r;
  } test_vec[8] = {
    { simde_x_vload_u16(UINT16_C(53542), UINT16_C(57259), UINT16_C(22339), UINT16_C(12715)),
      simde_x_vload_u16(UINT16_C(42872), UINT16_C(37334), UINT16_C(53927), UINT16_C(44623)),
      simde_x_vload_u16(UINT16_C(37185), UINT16_C(28367), UINT16_C(52012), UINT16_C( 7782)),
      simde_x_vload_u16(UINT16_C(47855), UINT16_C(48292), UINT16_C(20424), UINT16_C(41410)) },
    { simde_x_vload_u16(UINT16_C(22217), UINT16_C( 4033), UINT16_C(13479), UINT16_C(17229)),
      simde_x_vload_u16(UINT16_C(53798), UINT16_C(10199), UINT16_C( 6843), UINT16_C(  142)),
      simde_x_vload_u16(UINT16_C(25379), UINT16_C(64669), UINT16_C(23809), UINT16_C(41512)),
      simde_x_vload_u16(UINT16_C(59334), UINT16_C(58503), UINT16_C(30445), UINT16_C(58599)) },
    { simde_x_vload_u16(UINT16_C( 4818), UINT16_C(43879), UINT16_C(53895), UINT16_C(34932)),
      simde_x_vload_u16(UINT16_C(58386), UINT16_C(51971), UINT16_C(46830), UINT16_C(44331)),
      simde_x_vload_u16(UINT16_C(17659), UINT16_C(10340), UINT16_C( 6458), UINT16_C(33928)),
      simde_x_vload_u16(UINT16_C(29627), UINT16_C( 2248), UINT16_C(13523), UINT16_C(24529)) },
    { simde_x_vload_u16(UINT16_C(43316), UINT16_C(44719), UINT16_C(51823), UINT16_C(43333)),
      simde_x_vload_u16(UINT16_C(31630), UINT16_C(55632), UINT16_C(39907), UINT16_C(50392)),
      simde_x_vload_u16(UINT16_C( 6165), UINT16_C(24062), UINT16_C( 4142), UINT16_C( 5980)),
      simde_x_vload_u16(UINT16_C(17851), UINT16_C(13149), UINT16_C(16058), UINT16_C(64457)) },
    { simde_x_vload_u16(UINT16_C(24735), UINT16_C( 2400), UINT16_C(27653), UINT16_C(17049)),
      simde_x_vload_u16(UINT16_C(22251), UINT16_C(54387), UINT16_C(58984), UINT16_C(42034)),
      simde_x_vload_u16(UINT16_C(55851), UINT16_C(   54), UINT16_C( 5910), UINT16_C(63580)),
      simde_x_vload_u16(UINT16_C(58335), UINT16_C(13603), UINT16_C(40115), UINT16_C(38595)) },
    { simde_x_vload_u16(UINT16_C(15429), UINT16_C(50516), UINT16_C(30956), UINT16_C( 6870)),
      simde_x_vload_u16(UINT16_C(13012), UINT16_C( 3742), UINT16_C(54296), UINT16_C(39629)),
      simde_x_vload_u16(UINT16_C(13626), UINT16_C( 6485), UINT16_C(25089), UINT16_C(47775)),
      simde_x_vload_u16(UINT16_C(16043), UINT16_C(53259), UINT16_C( 1749), UINT16_C(15016)) },
    { simde_x_vload_u16(UINT16_C(38111), UINT16_C(62820), UINT16_C(44334), UINT16_C( 6797)),
      simde_x_vload_u16(UINT16_C( 2427), UINT16_C( 5948), UINT16_C(44340), UINT16_C( 2613)),
      simde_x_vload_u16(UINT16_C( 5569), UINT16_C(26888), UINT16_C( 1736), UINT16_C( 5668)),
      simde_x_vload_u16(UINT16_C(41253), UINT16_C(18224), UINT16_C( 1730), UINT16_C( 9852)) },
    { simde_x_vload_u16(UINT16_C(31418), UINT16_C(53439), UINT16_C(34735), UINT16_C(17073)),
      simde_x_vload_u16(UINT16_C(34974), UINT16_C(61606), UINT16_C(46368), UINT16_C(56165)),
      simde_x_vload_u16(UINT16_C(28056), UINT16_C(26996), UINT16_C(35828), UINT16_C(16558)),
      simde_x_vload_u16(UINT16_C(24500), UINT16_C(18829), UINT16_C(24195), UINT16_C(43002)) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_uint16x4_t a, b, c, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */
  /*   munit_rand_memory(sizeof(c), (uint8_t*) &c); */

  /*   r = simde_vaba_u16(a, b, c); */

  /*   printf("    { simde_x_vload_u16(UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 ")),\n", */
  /*          a.u16[0], a.u16[1], a.u16[2], a.u16[3]); */
  /*   printf("      simde_x_vload_u16(UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 ")),\n", */
  /*          b.u16[0], b.u16[1], b.u16[2], b.u16[3]); */
  /*   printf("      simde_x_vload_u16(UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 ")),\n", */
  /*          c.u16[0], c.u16[1], c.u16[2], c.u16[3]); */
  /*   printf("      simde_x_vload_u16(UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 ")) },\n", */
  /*          r.u16[0], r.u16[1], r.u16[2], r.u16[3]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint16x4_t r = simde_vaba_u16(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    simde_neon_assert_uint16x4(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vaba_u32(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_uint32x2_t a;
    simde_uint32x2_t b;
    simde_uint32x2_t c;
    simde_uint32x2_t r;
  } test_vec[8] = {
    { simde_x_vload_u32(UINT32_C(3752579366), UINT32_C( 833312579)),
      simde_x_vload_u32(UINT32_C(2446763896), UINT32_C(2924466855)),
      simde_x_vload_u32(UINT32_C(1859096897), UINT32_C( 510053164)),
      simde_x_vload_u32(UINT32_C(3164912367), UINT32_C(2713866184)) },
    { simde_x_vload_u32(UINT32_C( 264328905), UINT32_C(1129133223)),
      simde_x_vload_u32(UINT32_C( 668455462), UINT32_C(   9312955)),
      simde_x_vload_u32(UINT32_C(4238172963), UINT32_C(2720554241)),
      simde_x_vload_u32(UINT32_C(3834046406), UINT32_C(3840374509)) },
    { simde_x_vload_u32(UINT32_C(2875658962), UINT32_C(2289357447)),
      simde_x_vload_u32(UINT32_C(3406029842), UINT32_C(2905323246)),
      simde_x_vload_u32(UINT32_C( 677659899), UINT32_C(2223511866)),
      simde_x_vload_u32(UINT32_C( 147289019), UINT32_C(1607546067)) },
    { simde_x_vload_u32(UINT32_C(2930747700), UINT32_C(2839923311)),
      simde_x_vload_u32(UINT32_C(3645930382), UINT32_C(3302530019)),
      simde_x_vload_u32(UINT32_C(1576933397), UINT32_C( 391909422)),
      simde_x_vload_u32(UINT32_C( 861750715), UINT32_C(4224270010)) },
    { simde_x_vload_u32(UINT32_C( 157311135), UINT32_C(1117350917)),
      simde_x_vload_u32(UINT32_C(3564328683), UINT32_C(2754799208)),
      simde_x_vload_u32(UINT32_C(   3594795), UINT32_C(4166784790)),
      simde_x_vload_u32(UINT32_C( 891544543), UINT32_C(2529336499)) },
    { simde_x_vload_u32(UINT32_C(3310632005), UINT32_C( 450263276)),
      simde_x_vload_u32(UINT32_C( 245248724), UINT32_C(2597180440)),
      simde_x_vload_u32(UINT32_C( 425014586), UINT32_C(3131007489)),
      simde_x_vload_u32(UINT32_C(3490397867), UINT32_C( 984090325)) },
    { simde_x_vload_u32(UINT32_C(4117009631), UINT32_C( 445492526)),
      simde_x_vload_u32(UINT32_C( 389810555), UINT32_C( 171289908)),
      simde_x_vload_u32(UINT32_C(1762137537), UINT32_C( 371459784)),
      simde_x_vload_u32(UINT32_C(1194369317), UINT32_C( 645662402)) },
    { simde_x_vload_u32(UINT32_C(3502209722), UINT32_C(1118930863)),
      simde_x_vload_u32(UINT32_C(4037445790), UINT32_C(3680875808)),
      simde_x_vload_u32(UINT32_C(1769237912), UINT32_C(1085180916)),
      simde_x_vload_u32(UINT32_C(1234001844), UINT32_C(2818203267)) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_uint32x2_t a, b, c, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */
  /*   munit_rand_memory(sizeof(c), (uint8_t*) &c); */

  /*   r = simde_vaba_u32(a, b, c); */

  /*   printf("    { simde_x_vload_u32(UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 ")),\n", */
  /*          a.u32[0], a.u32[1]); */
  /*   printf("      simde_x_vload_u32(UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 ")),\n", */
  /*          b.u32[0], b.u32[1]); */
  /*   printf("      simde_x_vload_u32(UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 ")),\n", */
  /*          c.u32[0], c.u32[1]); */
  /*   printf("      simde_x_vload_u32(UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 ")) },\n", */
  /*          r.u32[0], r.u32[1]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint32x2_t r = simde_vaba_u32(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    simde_neon_assert_uint32x2(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabaq_s8(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int8x16_t a;
    simde_int8x16_t b;
    simde_int8x16_t c;
    simde_int8x16_t r;
  } test_vec[8] = {
    { simde_x_vloadq_s8(INT8_C(  38), INT8_C( -47), INT8_C( -85), INT8_C( -33),
                        INT8_C(  67), INT8_C(  87), INT8_C( -85), INT8_C(  49),
                        INT8_C( 120), INT8_C( -89), INT8_C( -42), INT8_C(-111),
                        INT8_C( -89), INT8_C( -46), INT8_C(  79), INT8_C( -82)),
      simde_x_vloadq_s8(INT8_C(  65), INT8_C(-111), INT8_C( -49), INT8_C( 110),
                        INT8_C(  44), INT8_C( -53), INT8_C( 102), INT8_C(  30),
                        INT8_C( -55), INT8_C(  86), INT8_C( -63), INT8_C(  15),
                        INT8_C( -89), INT8_C(  52), INT8_C(  77), INT8_C(  67)),
      simde_x_vloadq_s8(INT8_C(  38), INT8_C( -46), INT8_C( -41), INT8_C(  39),
                        INT8_C( -69), INT8_C(  26), INT8_C(-114), INT8_C(   0),
                        INT8_C(  35), INT8_C(  99), INT8_C( -99), INT8_C(  -4),
                        INT8_C(   1), INT8_C(  93), INT8_C(  40), INT8_C( -94)),
      simde_x_vloadq_s8(INT8_C(  65), INT8_C(  18), INT8_C( -77), INT8_C(  38),
                        INT8_C( -76), INT8_C( -90), INT8_C(-125), INT8_C(  79),
                        INT8_C( -46), INT8_C( -76), INT8_C(  -6), INT8_C( -92),
                        INT8_C(   1), INT8_C(  -5), INT8_C( 116), INT8_C(  79)) },
    { simde_x_vloadq_s8(INT8_C( -46), INT8_C(  18), INT8_C( 103), INT8_C( -85),
                        INT8_C(-121), INT8_C( -46), INT8_C( 116), INT8_C(-120),
                        INT8_C(  18), INT8_C( -28), INT8_C(   3), INT8_C( -53),
                        INT8_C( -18), INT8_C( -74), INT8_C(  43), INT8_C( -83)),
      simde_x_vloadq_s8(INT8_C(  -5), INT8_C(  68), INT8_C( 100), INT8_C(  40),
                        INT8_C(  58), INT8_C(  25), INT8_C(-120), INT8_C(-124),
                        INT8_C(  52), INT8_C( -87), INT8_C( -81), INT8_C( -82),
                        INT8_C( 111), INT8_C( -54), INT8_C(  69), INT8_C( -87)),
      simde_x_vloadq_s8(INT8_C(-114), INT8_C( 123), INT8_C(  80), INT8_C( -39),
                        INT8_C( -29), INT8_C(-101), INT8_C( -40), INT8_C( -60),
                        INT8_C(  21), INT8_C(  24), INT8_C(  -2), INT8_C(  93),
                        INT8_C(  46), INT8_C(  16), INT8_C(  92), INT8_C(  23)),
      simde_x_vloadq_s8(INT8_C(  63), INT8_C(  73), INT8_C( 123), INT8_C(  -6),
                        INT8_C( -34), INT8_C(  80), INT8_C( -60), INT8_C( -56),
                        INT8_C(  49), INT8_C(  83), INT8_C(  82), INT8_C( 122),
                        INT8_C(  47), INT8_C(  -4), INT8_C(  66), INT8_C(  27)) },
    { simde_x_vloadq_s8(INT8_C( -97), INT8_C(  96), INT8_C(  96), INT8_C(   9),
                        INT8_C(   5), INT8_C( 108), INT8_C(-103), INT8_C(  66),
                        INT8_C( -21), INT8_C(  86), INT8_C( 115), INT8_C( -44),
                        INT8_C( 104), INT8_C( -26), INT8_C(  50), INT8_C( -92)),
      simde_x_vloadq_s8(INT8_C(  43), INT8_C( -38), INT8_C(  54), INT8_C(   0),
                        INT8_C(  22), INT8_C(  23), INT8_C(  92), INT8_C(  -8),
                        INT8_C(  69), INT8_C(  60), INT8_C(  84), INT8_C( -59),
                        INT8_C( -20), INT8_C( 120), INT8_C( -42), INT8_C(  26)),
      simde_x_vloadq_s8(INT8_C( -44), INT8_C(  50), INT8_C( -98), INT8_C(  14),
                        INT8_C(  24), INT8_C( -44), INT8_C( -51), INT8_C(-102),
                        INT8_C(  58), INT8_C(  53), INT8_C(  85), INT8_C(  25),
                        INT8_C(   1), INT8_C(  98), INT8_C( -97), INT8_C( -70)),
      simde_x_vloadq_s8(INT8_C( -10), INT8_C( -72), INT8_C(  -8), INT8_C(  23),
                        INT8_C(   7), INT8_C( -81), INT8_C(  40), INT8_C( -96),
                        INT8_C( -10), INT8_C(  93), INT8_C( 116), INT8_C(  40),
                        INT8_C( 125), INT8_C(  -4), INT8_C( 105), INT8_C(   4)) },
    { simde_x_vloadq_s8(INT8_C( -33), INT8_C(-108), INT8_C( 100), INT8_C( -11),
                        INT8_C(  46), INT8_C( -83), INT8_C(-115), INT8_C(  26),
                        INT8_C( 123), INT8_C(   9), INT8_C(  60), INT8_C(  23),
                        INT8_C(  52), INT8_C( -83), INT8_C(  53), INT8_C(  10)),
      simde_x_vloadq_s8(INT8_C( -63), INT8_C(  21), INT8_C(   8), INT8_C( 105),
                        INT8_C( -56), INT8_C(   6), INT8_C(  36), INT8_C(  22),
                        INT8_C( -70), INT8_C( 122), INT8_C( -65), INT8_C( -48),
                        INT8_C( -81), INT8_C(-121), INT8_C( -79), INT8_C(  66)),
      simde_x_vloadq_s8(INT8_C( -98), INT8_C(-120), INT8_C( -90), INT8_C( -16),
                        INT8_C(  32), INT8_C( -75), INT8_C( 101), INT8_C( -37),
                        INT8_C(-104), INT8_C( 109), INT8_C( 116), INT8_C( 105),
                        INT8_C( -12), INT8_C(-117), INT8_C( -82), INT8_C(  64)),
      simde_x_vloadq_s8(INT8_C(   2), INT8_C(  33), INT8_C( -58), INT8_C( 110),
                        INT8_C(-122), INT8_C(  -2), INT8_C( -50), INT8_C(  85),
                        INT8_C( -99), INT8_C(  22), INT8_C( -15), INT8_C( -80),
                        INT8_C( 121), INT8_C( -79), INT8_C(  56), INT8_C(  12)) },
    { simde_x_vloadq_s8(INT8_C(  57), INT8_C(  80), INT8_C( 111), INT8_C(  19),
                        INT8_C(  71), INT8_C(  51), INT8_C(  67), INT8_C( -67),
                        INT8_C(  31), INT8_C( -82), INT8_C(-102), INT8_C( -52),
                        INT8_C( 125), INT8_C( -25), INT8_C( -99), INT8_C( -91)),
      simde_x_vloadq_s8(INT8_C(  90), INT8_C(  26), INT8_C(  77), INT8_C( -56),
                        INT8_C(-107), INT8_C( -70), INT8_C( -33), INT8_C(  50),
                        INT8_C(  31), INT8_C( -66), INT8_C( -47), INT8_C(-100),
                        INT8_C(  94), INT8_C( 117), INT8_C(  -2), INT8_C( 100)),
      simde_x_vloadq_s8(INT8_C(  86), INT8_C(-102), INT8_C( 120), INT8_C(  75),
                        INT8_C(  10), INT8_C( 116), INT8_C(  92), INT8_C( -12),
                        INT8_C( -78), INT8_C(  91), INT8_C(  27), INT8_C(-126),
                        INT8_C( -20), INT8_C(-127), INT8_C(  -8), INT8_C(   6)),
      simde_x_vloadq_s8(INT8_C(  61), INT8_C( -48), INT8_C(-102), INT8_C(-106),
                        INT8_C( -68), INT8_C( -19), INT8_C( -64), INT8_C(  -5),
                        INT8_C(-116), INT8_C(  75), INT8_C( -28), INT8_C( -26),
                        INT8_C( -17), INT8_C( -37), INT8_C( -93), INT8_C(   3)) },
    { simde_x_vloadq_s8(INT8_C(  74), INT8_C( 103), INT8_C(-119), INT8_C(  56),
                        INT8_C( -39), INT8_C(   9), INT8_C( -35), INT8_C(  15),
                        INT8_C(   3), INT8_C(  20), INT8_C(  35), INT8_C( -91),
                        INT8_C(  -7), INT8_C( 119), INT8_C(  37), INT8_C( 116)),
      simde_x_vloadq_s8(INT8_C(  50), INT8_C(  10), INT8_C(  73), INT8_C(   6),
                        INT8_C( -55), INT8_C(  82), INT8_C( 117), INT8_C(  67),
                        INT8_C( -34), INT8_C( -65), INT8_C(  51), INT8_C( -86),
                        INT8_C( 124), INT8_C(-115), INT8_C( -15), INT8_C(   0)),
      simde_x_vloadq_s8(INT8_C(  20), INT8_C( -76), INT8_C( -73), INT8_C(  99),
                        INT8_C( 100), INT8_C(-101), INT8_C(  30), INT8_C(  68),
                        INT8_C( 115), INT8_C( -98), INT8_C(-115), INT8_C( 105),
                        INT8_C( -14), INT8_C(-114), INT8_C(  33), INT8_C( -79)),
      simde_x_vloadq_s8(INT8_C( 104), INT8_C( -67), INT8_C(  27), INT8_C(-107),
                        INT8_C( 116), INT8_C( -64), INT8_C(  52), INT8_C(  16),
                        INT8_C(-104), INT8_C(  53), INT8_C( -55), INT8_C( 100),
                        INT8_C(-125), INT8_C( 120), INT8_C(  85), INT8_C( -61)) },
    { simde_x_vloadq_s8(INT8_C( -36), INT8_C( -82), INT8_C( -64), INT8_C( -41),
                        INT8_C( -75), INT8_C(-125), INT8_C(-101), INT8_C(  27),
                        INT8_C( 108), INT8_C(   0), INT8_C( 102), INT8_C( -31),
                        INT8_C(  47), INT8_C(   7), INT8_C( -93), INT8_C(  30)),
      simde_x_vloadq_s8(INT8_C( -13), INT8_C(  93), INT8_C( -44), INT8_C( -19),
                        INT8_C( -11), INT8_C( -65), INT8_C( -14), INT8_C( -99),
                        INT8_C(-106), INT8_C( -64), INT8_C(  58), INT8_C( 113),
                        INT8_C( -89), INT8_C(  48), INT8_C( -71), INT8_C(  -3)),
      simde_x_vloadq_s8(INT8_C(  17), INT8_C(  -5), INT8_C( 110), INT8_C(  34),
                        INT8_C(  20), INT8_C( -41), INT8_C( 109), INT8_C( -48),
                        INT8_C(-115), INT8_C(  49), INT8_C(  95), INT8_C( -75),
                        INT8_C(   7), INT8_C(  -1), INT8_C( -59), INT8_C(  82)),
      simde_x_vloadq_s8(INT8_C(  -6), INT8_C(  16), INT8_C(  90), INT8_C(  12),
                        INT8_C( -44), INT8_C(-101), INT8_C(  22), INT8_C(  78),
                        INT8_C( 117), INT8_C( 113), INT8_C(-117), INT8_C( -99),
                        INT8_C(-113), INT8_C(  56), INT8_C( -81), INT8_C( 115)) },
    { simde_x_vloadq_s8(INT8_C(  91), INT8_C(  35), INT8_C( -18), INT8_C( -31),
                        INT8_C( -38), INT8_C(  99), INT8_C(  58), INT8_C( -36),
                        INT8_C( -91), INT8_C(-112), INT8_C( -88), INT8_C(  10),
                        INT8_C(  99), INT8_C(   0), INT8_C(  -6), INT8_C( -74)),
      simde_x_vloadq_s8(INT8_C(  12), INT8_C(  56), INT8_C(  19), INT8_C(  34),
                        INT8_C(  38), INT8_C(  45), INT8_C(  -6), INT8_C(  90),
                        INT8_C(  71), INT8_C(-120), INT8_C( -56), INT8_C(  79),
                        INT8_C(-110), INT8_C(  57), INT8_C( -52), INT8_C(  87)),
      simde_x_vloadq_s8(INT8_C(  10), INT8_C( -53), INT8_C( 127), INT8_C(  61),
                        INT8_C(-115), INT8_C( -28), INT8_C( 124), INT8_C(  45),
                        INT8_C( 118), INT8_C(  53), INT8_C( -42), INT8_C(-122),
                        INT8_C( 120), INT8_C( -79), INT8_C(  -7), INT8_C(-104)),
      simde_x_vloadq_s8(INT8_C(  93), INT8_C(-112), INT8_C(  90), INT8_C(  -4),
                        INT8_C( 115), INT8_C( -84), INT8_C( -68), INT8_C(   9),
                        INT8_C( -44), INT8_C(  61), INT8_C( -74), INT8_C( -45),
                        INT8_C(  73), INT8_C(-120), INT8_C(  39), INT8_C( 117)) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_int8x16_t a, b, c, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */
  /*   munit_rand_memory(sizeof(c), (uint8_t*) &c); */

  /*   r = simde_vabaq_s8(a, b, c); */

  /*   printf("    { simde_x_vloadq_s8(INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 ")),\n", */
  /*          a.i8[ 0], a.i8[ 1], a.i8[ 2], a.i8[ 3], a.i8[ 4], a.i8[ 5], a.i8[ 6], a.i8[ 7], */
  /*          a.i8[ 8], a.i8[ 9], a.i8[10], a.i8[11], a.i8[12], a.i8[13], a.i8[14], a.i8[15]); */
  /*   printf("      simde_x_vloadq_s8(INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 ")),\n", */
  /*          b.i8[ 0], b.i8[ 1], b.i8[ 2], b.i8[ 3], b.i8[ 4], b.i8[ 5], b.i8[ 6], b.i8[ 7], */
  /*          b.i8[ 8], b.i8[ 9], b.i8[10], b.i8[11], b.i8[12], b.i8[13], b.i8[14], b.i8[15]); */
  /*   printf("      simde_x_vloadq_s8(INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 ")),\n", */
  /*          c.i8[ 0], c.i8[ 1], c.i8[ 2], c.i8[ 3], c.i8[ 4], c.i8[ 5], c.i8[ 6], c.i8[ 7], */
  /*          c.i8[ 8], c.i8[ 9], c.i8[10], c.i8[11], c.i8[12], c.i8[13], c.i8[14], c.i8[15]); */
  /*   printf("      simde_x_vloadq_s8(INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 ")) },\n", */
  /*          r.i8[ 0], r.i8[ 1], r.i8[ 2], r.i8[ 3], r.i8[ 4], r.i8[ 5], r.i8[ 6], r.i8[ 7], */
  /*          r.i8[ 8], r.i8[ 9], r.i8[10], r.i8[11], r.i8[12], r.i8[13], r.i8[14], r.i8[15]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int8x16_t r = simde_vabaq_s8(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    simde_neon_assert_int8x16(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabaq_s16(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int16x8_t a;
    simde_int16x8_t b;
    simde_int16x8_t c;
    simde_int16x8_t r;
  } test_vec[8] = {
    { simde_x_vloadq_s16(INT16_C(-11994), INT16_C( -8277), INT16_C( 22339), INT16_C( 12715),
                         INT16_C(-22664), INT16_C(-28202), INT16_C(-11609), INT16_C(-20913)),
      simde_x_vloadq_s16(INT16_C(-28351), INT16_C( 28367), INT16_C(-13524), INT16_C(  7782),
                         INT16_C( 22217), INT16_C(  4033), INT16_C( 13479), INT16_C( 17229)),
      simde_x_vloadq_s16(INT16_C(-11738), INT16_C( 10199), INT16_C(  6843), INT16_C(   142),
                         INT16_C( 25379), INT16_C(  -867), INT16_C( 23809), INT16_C(-24024)),
      simde_x_vloadq_s16(INT16_C(  4619), INT16_C(  9891), INT16_C(-22830), INT16_C( 20355),
                         INT16_C(-19502), INT16_C(-23302), INT16_C( -1279), INT16_C( 20340)) },
    { simde_x_vloadq_s16(INT16_C(  4818), INT16_C(-21657), INT16_C(-11641), INT16_C(-30604),
                         INT16_C( -7150), INT16_C(-13565), INT16_C(-18706), INT16_C(-21205)),
      simde_x_vloadq_s16(INT16_C( 17659), INT16_C( 10340), INT16_C(  6458), INT16_C(-31608),
                         INT16_C(-22220), INT16_C(-20817), INT16_C(-13713), INT16_C(-22203)),
      simde_x_vloadq_s16(INT16_C( 31630), INT16_C( -9904), INT16_C(-25629), INT16_C(-15144),
                         INT16_C(  6165), INT16_C( 24062), INT16_C(  4142), INT16_C(  5980)),
      simde_x_vloadq_s16(INT16_C( 18789), INT16_C( -1413), INT16_C( 20446), INT16_C(-14140),
                         INT16_C( 21235), INT16_C( 31314), INT16_C(  -851), INT16_C(  6978)) },
    { simde_x_vloadq_s16(INT16_C( 24735), INT16_C(  2400), INT16_C( 27653), INT16_C( 17049),
                         INT16_C( 22251), INT16_C(-11149), INT16_C( -6552), INT16_C(-23502)),
      simde_x_vloadq_s16(INT16_C( -9685), INT16_C(    54), INT16_C(  5910), INT16_C( -1956),
                         INT16_C( 15429), INT16_C(-15020), INT16_C( 30956), INT16_C(  6870)),
      simde_x_vloadq_s16(INT16_C( 13012), INT16_C(  3742), INT16_C(-11240), INT16_C(-25907),
                         INT16_C( 13626), INT16_C(  6485), INT16_C( 25089), INT16_C(-17761)),
      simde_x_vloadq_s16(INT16_C(-18104), INT16_C(  6088), INT16_C(-20733), INT16_C(-24536),
                         INT16_C( 24054), INT16_C( 10356), INT16_C(  -685), INT16_C(  1129)) },
    { simde_x_vloadq_s16(INT16_C(-27425), INT16_C( -2716), INT16_C(-21202), INT16_C(  6797),
                         INT16_C(  2427), INT16_C(  5948), INT16_C(-21196), INT16_C(  2613)),
      simde_x_vloadq_s16(INT16_C(  5569), INT16_C( 26888), INT16_C(  1736), INT16_C(  5668),
                         INT16_C( 31418), INT16_C(-12097), INT16_C(-30801), INT16_C( 17073)),
      simde_x_vloadq_s16(INT16_C(-30562), INT16_C( -3930), INT16_C(-19168), INT16_C( -9371),
                         INT16_C( 28056), INT16_C( 26996), INT16_C(-29708), INT16_C( 16558)),
      simde_x_vloadq_s16(INT16_C(  8706), INT16_C( 28102), INT16_C(  -298), INT16_C( 21836),
                         INT16_C(  5789), INT16_C(-20495), INT16_C(-20103), INT16_C(  3128)) },
    { simde_x_vloadq_s16(INT16_C( 20537), INT16_C(  4975), INT16_C( 13127), INT16_C(-17085),
                         INT16_C(-20961), INT16_C(-13158), INT16_C( -6275), INT16_C(-23139)),
      simde_x_vloadq_s16(INT16_C(  6746), INT16_C(-14259), INT16_C(-17771), INT16_C( 13023),
                         INT16_C(-16865), INT16_C(-25391), INT16_C( 30046), INT16_C( 25854)),
      simde_x_vloadq_s16(INT16_C(-26026), INT16_C( 19320), INT16_C( 29706), INT16_C( -2980),
                         INT16_C( 23474), INT16_C(-32229), INT16_C(-32276), INT16_C(  1784)),
      simde_x_vloadq_s16(INT16_C(-12227), INT16_C(-26982), INT16_C( -4932), INT16_C( -1082),
                         INT16_C( 19378), INT16_C( -6320), INT16_C( -9489), INT16_C(   931)) },
    { simde_x_vloadq_s16(INT16_C( 26442), INT16_C( 14473), INT16_C(  2521), INT16_C(  4061),
                         INT16_C(  5123), INT16_C(-23261), INT16_C( 30713), INT16_C( 29733)),
      simde_x_vloadq_s16(INT16_C(  2610), INT16_C(  1609), INT16_C( 21193), INT16_C( 17269),
                         INT16_C(-16418), INT16_C(-21965), INT16_C(-29316), INT16_C(   241)),
      simde_x_vloadq_s16(INT16_C(-19436), INT16_C( 25527), INT16_C(-25756), INT16_C( 17438),
                         INT16_C(-24973), INT16_C( 27021), INT16_C(-28942), INT16_C(-20191)),
      simde_x_vloadq_s16(INT16_C(-17048), INT16_C(-27145), INT16_C(-16066), INT16_C(  4230),
                         INT16_C( 13678), INT16_C( 25725), INT16_C( 31087), INT16_C(-15371)) },
    { simde_x_vloadq_s16(INT16_C(-20772), INT16_C(-10304), INT16_C(-31819), INT16_C(  7067),
                         INT16_C(   108), INT16_C( -7834), INT16_C(  1839), INT16_C(  7843)),
      simde_x_vloadq_s16(INT16_C( 24051), INT16_C( -4652), INT16_C(-16395), INT16_C(-25102),
                         INT16_C(-16234), INT16_C( 28986), INT16_C( 12455), INT16_C(  -583)),
      simde_x_vloadq_s16(INT16_C( -1263), INT16_C(  8814), INT16_C(-10476), INT16_C(-12179),
                         INT16_C( 12685), INT16_C(-19105), INT16_C(  -249), INT16_C( 21189)),
      simde_x_vloadq_s16(INT16_C(  4542), INT16_C(  3162), INT16_C(-25900), INT16_C( 19990),
                         INT16_C( 29027), INT16_C(-25279), INT16_C( 14543), INT16_C( 29615)) },
    { simde_x_vloadq_s16(INT16_C(  9051), INT16_C( -7698), INT16_C( 25562), INT16_C( -9158),
                         INT16_C(-28507), INT16_C(  2728), INT16_C(    99), INT16_C(-18694)),
      simde_x_vloadq_s16(INT16_C( 14348), INT16_C(  8723), INT16_C( 11558), INT16_C( 23290),
                         INT16_C(-30649), INT16_C( 20424), INT16_C( 14738), INT16_C( 22476)),
      simde_x_vloadq_s16(INT16_C(-13558), INT16_C( 15743), INT16_C( -7027), INT16_C( 11644),
                         INT16_C( 13686), INT16_C(-31018), INT16_C(-20104), INT16_C(-26375)),
      simde_x_vloadq_s16(INT16_C(-28579), INT16_C(  -678), INT16_C(-21389), INT16_C(  2488),
                         INT16_C( 15828), INT16_C(-11366), INT16_C(-30595), INT16_C( 30157)) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_int16x8_t a, b, c, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */
  /*   munit_rand_memory(sizeof(c), (uint8_t*) &c); */

  /*   r = simde_vabaq_s16(a, b, c); */

  /*   printf("    { simde_x_vloadq_s16(INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "),\n" */
  /*          "                         INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 ")),\n", */
  /*          a.i16[0], a.i16[1], a.i16[2], a.i16[3], a.i16[4], a.i16[5], a.i16[6], a.i16[7]); */
  /*   printf("      simde_x_vloadq_s16(INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "),\n" */
  /*          "                         INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 ")),\n", */
  /*          b.i16[0], b.i16[1], b.i16[2], b.i16[3], b.i16[4], b.i16[5], b.i16[6], b.i16[7]); */
  /*   printf("      simde_x_vloadq_s16(INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "),\n" */
  /*          "                         INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 ")),\n", */
  /*          c.i16[0], c.i16[1], c.i16[2], c.i16[3], c.i16[4], c.i16[5], c.i16[6], c.i16[7]); */
  /*   printf("      simde_x_vloadq_s16(INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "),\n" */
  /*          "                         INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 ")) },\n", */
  /*          r.i16[0], r.i16[1], r.i16[2], r.i16[3], r.i16[4], r.i16[5], r.i16[6], r.i16[7]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int16x8_t r = simde_vabaq_s16(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    simde_neon_assert_int16x8(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabaq_s32(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int32x4_t a;
    simde_int32x4_t b;
    simde_int32x4_t c;
    simde_int32x4_t r;
  } test_vec[8] = {
    { simde_x_vloadq_s32(INT32_C( -542387930), INT32_C(  833312579),
                         INT32_C(-1848203400), INT32_C(-1370500441)),
      simde_x_vloadq_s32(INT32_C( 1859096897), INT32_C(  510053164),
                         INT32_C(  264328905), INT32_C( 1129133223)),
      simde_x_vloadq_s32(INT32_C(  668455462), INT32_C(    9312955),
                         INT32_C(  -56794333), INT32_C(-1574413055)),
      simde_x_vloadq_s32(INT32_C(  648253505), INT32_C( 1334052788),
                         INT32_C(-1527080162), INT32_C(  220920577)) },
    { simde_x_vloadq_s32(INT32_C(-1419308334), INT32_C(-2005609849),
                         INT32_C( -888937454), INT32_C(-1389644050)),
      simde_x_vloadq_s32(INT32_C(  677659899), INT32_C(-2071455430),
                         INT32_C(-1364219596), INT32_C(-1455043985)),
      simde_x_vloadq_s32(INT32_C( -649036914), INT32_C( -992437277),
                         INT32_C( 1576933397), INT32_C(  391909422)),
      simde_x_vloadq_s32(INT32_C(  -92611521), INT32_C( -926591696),
                         INT32_C(  464876849), INT32_C(  457309357)) },
    { simde_x_vloadq_s32(INT32_C(  157311135), INT32_C( 1117350917),
                         INT32_C( -730638613), INT32_C(-1540168088)),
      simde_x_vloadq_s32(INT32_C(    3594795), INT32_C( -128182506),
                         INT32_C( -984335291), INT32_C(  450263276)),
      simde_x_vloadq_s32(INT32_C(  245248724), INT32_C(-1697786856),
                         INT32_C(  425014586), INT32_C(-1163959807)),
      simde_x_vloadq_s32(INT32_C(  398965064), INT32_C(-1608012029),
                         INT32_C(  678711264), INT32_C(   74054995)) },
    { simde_x_vloadq_s32(INT32_C( -177957665), INT32_C(  445492526),
                         INT32_C(  389810555), INT32_C(  171289908)),
      simde_x_vloadq_s32(INT32_C( 1762137537), INT32_C(  371459784),
                         INT32_C( -792757574), INT32_C( 1118930863)),
      simde_x_vloadq_s32(INT32_C( -257521506), INT32_C( -614091488),
                         INT32_C( 1769237912), INT32_C( 1085180916)),
      simde_x_vloadq_s32(INT32_C( 1841701378), INT32_C( 1431043798),
                         INT32_C( 2122782365), INT32_C(  205039855)) },
    { simde_x_vloadq_s32(INT32_C(  326062137), INT32_C(-1119669433),
                         INT32_C( -862278113), INT32_C(-1516378243)),
      simde_x_vloadq_s32(INT32_C( -934471078), INT32_C(  853523093),
                         INT32_C(-1663975905), INT32_C( 1694397790)),
      simde_x_vloadq_s32(INT32_C( 1266195030), INT32_C( -195267574),
                         INT32_C(-2112136270), INT32_C(  116949484)),
      simde_x_vloadq_s32(INT32_C(-1874603971), INT32_C(  -70878766),
                         INT32_C( -414117748), INT32_C(   61070063)) },
    { simde_x_vloadq_s32(INT32_C(  948528970), INT32_C(  266144217),
                         INT32_C(-1524427773), INT32_C( 1948612601)),
      simde_x_vloadq_s32(INT32_C(  105450034), INT32_C( 1131762377),
                         INT32_C(-1439449122), INT32_C(   15830396)),
      simde_x_vloadq_s32(INT32_C( 1672983572), INT32_C( 1142856548),
                         INT32_C( 1770888819), INT32_C(-1323200782)),
      simde_x_vloadq_s32(INT32_C(-1778904788), INT32_C(  277238388),
                         INT32_C( -439798418), INT32_C(-1007323517)) },
    { simde_x_vloadq_s32(INT32_C( -675238180), INT32_C(  463176629),
                         INT32_C( -513408916), INT32_C(  514000687)),
      simde_x_vloadq_s32(INT32_C( -304849421), INT32_C(-1645035531),
                         INT32_C( 1899675798), INT32_C(  -38195033)),
      simde_x_vloadq_s32(INT32_C(  577698577), INT32_C( -798107884),
                         INT32_C(-1252052595), INT32_C( 1388707591)),
      simde_x_vloadq_s32(INT32_C(  207309818), INT32_C( 1310104276),
                         INT32_C(  629829987), INT32_C( 1940903311)) },
    { simde_x_vloadq_s32(INT32_C( -504487077), INT32_C( -600153126),
                         INT32_C(  178819237), INT32_C(-1225129885)),
      simde_x_vloadq_s32(INT32_C(  571684876), INT32_C( 1526344998),
                         INT32_C( 1338542151), INT32_C( 1473001874)),
      simde_x_vloadq_s32(INT32_C( 1031785226), INT32_C(  763159693),
                         INT32_C(-2032781962), INT32_C(-1728466568)),
      simde_x_vloadq_s32(INT32_C(  -44386727), INT32_C(  163032179),
                         INT32_C( 1102462420), INT32_C( -131631031)) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_int32x4_t a, b, c, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */
  /*   munit_rand_memory(sizeof(c), (uint8_t*) &c); */

  /*   r = simde_vabaq_s32(a, b, c); */

  /*   printf("    { simde_x_vloadq_s32(INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 "),\n" */
  /*          "                         INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 ")),\n", */
  /*          a.i32[0], a.i32[1], a.i32[2], a.i32[3]); */
  /*   printf("      simde_x_vloadq_s32(INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 "),\n" */
  /*          "                         INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 ")),\n", */
  /*          b.i32[0], b.i32[1], b.i32[2], b.i32[3]); */
  /*   printf("      simde_x_vloadq_s32(INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 "),\n" */
  /*          "                         INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 ")),\n", */
  /*          c.i32[0], c.i32[1], c.i32[2], c.i32[3]); */
  /*   printf("      simde_x_vloadq_s32(INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 "),\n" */
  /*          "                         INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 ")) },\n", */
  /*          r.i32[0], r.i32[1], r.i32[2], r.i32[3]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int32x4_t r = simde_vabaq_s32(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    simde_neon_assert_int32x4(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabaq_u8(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_uint8x16_t a;
    simde_uint8x16_t b;
    simde_uint8x16_t c;
    simde_uint8x16_t r;
  } test_vec[8] = {
    { simde_x_vloadq_u8(UINT8_C( 38), UINT8_C(209), UINT8_C(171), UINT8_C(223),
                        UINT8_C( 67), UINT8_C( 87), UINT8_C(171), UINT8_C( 49),
                        UINT8_C(120), UINT8_C(167), UINT8_C(214), UINT8_C(145),
                        UINT8_C(167), UINT8_C(210), UINT8_C( 79), UINT8_C(174) ),
      simde_x_vloadq_u8(UINT8_C( 65), UINT8_C(145), UINT8_C(207), UINT8_C(110),
                        UINT8_C( 44), UINT8_C(203), UINT8_C(102), UINT8_C( 30),
                        UINT8_C(201), UINT8_C( 86), UINT8_C(193), UINT8_C( 15),
                        UINT8_C(167), UINT8_C( 52), UINT8_C( 77), UINT8_C( 67) ),
      simde_x_vloadq_u8(UINT8_C( 38), UINT8_C(210), UINT8_C(215), UINT8_C( 39),
                        UINT8_C(187), UINT8_C( 26), UINT8_C(142), UINT8_C(  0),
                        UINT8_C( 35), UINT8_C( 99), UINT8_C(157), UINT8_C(252),
                        UINT8_C(  1), UINT8_C( 93), UINT8_C( 40), UINT8_C(162) ),
      simde_x_vloadq_u8(UINT8_C( 11), UINT8_C( 18), UINT8_C(179), UINT8_C(152),
                        UINT8_C(210), UINT8_C(166), UINT8_C(211), UINT8_C( 19),
                        UINT8_C(210), UINT8_C(180), UINT8_C(178), UINT8_C(126),
                        UINT8_C(  1), UINT8_C(251), UINT8_C( 42), UINT8_C( 13) ) },
    { simde_x_vloadq_u8(UINT8_C(210), UINT8_C( 18), UINT8_C(103), UINT8_C(171),
                        UINT8_C(135), UINT8_C(210), UINT8_C(116), UINT8_C(136),
                        UINT8_C( 18), UINT8_C(228), UINT8_C(  3), UINT8_C(203),
                        UINT8_C(238), UINT8_C(182), UINT8_C( 43), UINT8_C(173) ),
      simde_x_vloadq_u8(UINT8_C(251), UINT8_C( 68), UINT8_C(100), UINT8_C( 40),
                        UINT8_C( 58), UINT8_C( 25), UINT8_C(136), UINT8_C(132),
                        UINT8_C( 52), UINT8_C(169), UINT8_C(175), UINT8_C(174),
                        UINT8_C(111), UINT8_C(202), UINT8_C( 69), UINT8_C(169) ),
      simde_x_vloadq_u8(UINT8_C(142), UINT8_C(123), UINT8_C( 80), UINT8_C(217),
                        UINT8_C(227), UINT8_C(155), UINT8_C(216), UINT8_C(196),
                        UINT8_C( 21), UINT8_C( 24), UINT8_C(254), UINT8_C( 93),
                        UINT8_C( 46), UINT8_C( 16), UINT8_C( 92), UINT8_C( 23) ),
      simde_x_vloadq_u8(UINT8_C(101), UINT8_C( 73), UINT8_C( 83), UINT8_C( 92),
                        UINT8_C( 48), UINT8_C( 84), UINT8_C(196), UINT8_C(200),
                        UINT8_C(243), UINT8_C( 83), UINT8_C( 82), UINT8_C(122),
                        UINT8_C(173), UINT8_C(252), UINT8_C( 66), UINT8_C( 27) ) },
    { simde_x_vloadq_u8(UINT8_C(159), UINT8_C( 96), UINT8_C( 96), UINT8_C(  9),
                        UINT8_C(  5), UINT8_C(108), UINT8_C(153), UINT8_C( 66),
                        UINT8_C(235), UINT8_C( 86), UINT8_C(115), UINT8_C(212),
                        UINT8_C(104), UINT8_C(230), UINT8_C( 50), UINT8_C(164) ),
      simde_x_vloadq_u8(UINT8_C( 43), UINT8_C(218), UINT8_C( 54), UINT8_C(  0),
                        UINT8_C( 22), UINT8_C( 23), UINT8_C( 92), UINT8_C(248),
                        UINT8_C( 69), UINT8_C( 60), UINT8_C( 84), UINT8_C(197),
                        UINT8_C(236), UINT8_C(120), UINT8_C(214), UINT8_C( 26) ),
      simde_x_vloadq_u8(UINT8_C(212), UINT8_C( 50), UINT8_C(158), UINT8_C( 14),
                        UINT8_C( 24), UINT8_C(212), UINT8_C(205), UINT8_C(154),
                        UINT8_C( 58), UINT8_C( 53), UINT8_C( 85), UINT8_C( 25),
                        UINT8_C(  1), UINT8_C( 98), UINT8_C(159), UINT8_C(186) ),
      simde_x_vloadq_u8(UINT8_C( 72), UINT8_C(184), UINT8_C(200), UINT8_C( 23),
                        UINT8_C(  7), UINT8_C( 41), UINT8_C( 10), UINT8_C(228),
                        UINT8_C(224), UINT8_C( 79), UINT8_C(116), UINT8_C( 40),
                        UINT8_C(125), UINT8_C(208), UINT8_C(251), UINT8_C( 68) ) },
    { simde_x_vloadq_u8(UINT8_C(223), UINT8_C(148), UINT8_C(100), UINT8_C(245),
                        UINT8_C( 46), UINT8_C(173), UINT8_C(141), UINT8_C( 26),
                        UINT8_C(123), UINT8_C(  9), UINT8_C( 60), UINT8_C( 23),
                        UINT8_C( 52), UINT8_C(173), UINT8_C( 53), UINT8_C( 10) ),
      simde_x_vloadq_u8(UINT8_C(193), UINT8_C( 21), UINT8_C(  8), UINT8_C(105),
                        UINT8_C(200), UINT8_C(  6), UINT8_C( 36), UINT8_C( 22),
                        UINT8_C(186), UINT8_C(122), UINT8_C(191), UINT8_C(208),
                        UINT8_C(175), UINT8_C(135), UINT8_C(177), UINT8_C( 66) ),
      simde_x_vloadq_u8(UINT8_C(158), UINT8_C(136), UINT8_C(166), UINT8_C(240),
                        UINT8_C( 32), UINT8_C(181), UINT8_C(101), UINT8_C(219),
                        UINT8_C(152), UINT8_C(109), UINT8_C(116), UINT8_C(105),
                        UINT8_C(244), UINT8_C(139), UINT8_C(174), UINT8_C( 64) ),
      simde_x_vloadq_u8(UINT8_C(188), UINT8_C(  7), UINT8_C(  2), UINT8_C(124),
                        UINT8_C(134), UINT8_C( 92), UINT8_C(206), UINT8_C(223),
                        UINT8_C( 89), UINT8_C(252), UINT8_C(241), UINT8_C(176),
                        UINT8_C(121), UINT8_C(177), UINT8_C( 50), UINT8_C(  8) ) },
    { simde_x_vloadq_u8(UINT8_C( 57), UINT8_C( 80), UINT8_C(111), UINT8_C( 19),
                        UINT8_C( 71), UINT8_C( 51), UINT8_C( 67), UINT8_C(189),
                        UINT8_C( 31), UINT8_C(174), UINT8_C(154), UINT8_C(204),
                        UINT8_C(125), UINT8_C(231), UINT8_C(157), UINT8_C(165) ),
      simde_x_vloadq_u8(UINT8_C( 90), UINT8_C( 26), UINT8_C( 77), UINT8_C(200),
                        UINT8_C(149), UINT8_C(186), UINT8_C(223), UINT8_C( 50),
                        UINT8_C( 31), UINT8_C(190), UINT8_C(209), UINT8_C(156),
                        UINT8_C( 94), UINT8_C(117), UINT8_C(254), UINT8_C(100) ),
      simde_x_vloadq_u8(UINT8_C( 86), UINT8_C(154), UINT8_C(120), UINT8_C( 75),
                        UINT8_C( 10), UINT8_C(116), UINT8_C( 92), UINT8_C(244),
                        UINT8_C(178), UINT8_C( 91), UINT8_C( 27), UINT8_C(130),
                        UINT8_C(236), UINT8_C(129), UINT8_C(248), UINT8_C(  6) ),
      simde_x_vloadq_u8(UINT8_C( 53), UINT8_C(208), UINT8_C(154), UINT8_C(150),
                        UINT8_C(188), UINT8_C(237), UINT8_C(192), UINT8_C(127),
                        UINT8_C(178), UINT8_C( 75), UINT8_C(228), UINT8_C(178),
                        UINT8_C( 11), UINT8_C(243), UINT8_C(151), UINT8_C( 71) ) },
    { simde_x_vloadq_u8(UINT8_C( 74), UINT8_C(103), UINT8_C(137), UINT8_C( 56),
                        UINT8_C(217), UINT8_C(  9), UINT8_C(221), UINT8_C( 15),
                        UINT8_C(  3), UINT8_C( 20), UINT8_C( 35), UINT8_C(165),
                        UINT8_C(249), UINT8_C(119), UINT8_C( 37), UINT8_C(116) ),
      simde_x_vloadq_u8(UINT8_C( 50), UINT8_C( 10), UINT8_C( 73), UINT8_C(  6),
                        UINT8_C(201), UINT8_C( 82), UINT8_C(117), UINT8_C( 67),
                        UINT8_C(222), UINT8_C(191), UINT8_C( 51), UINT8_C(170),
                        UINT8_C(124), UINT8_C(141), UINT8_C(241), UINT8_C(  0) ),
      simde_x_vloadq_u8(UINT8_C( 20), UINT8_C(180), UINT8_C(183), UINT8_C( 99),
                        UINT8_C(100), UINT8_C(155), UINT8_C( 30), UINT8_C( 68),
                        UINT8_C(115), UINT8_C(158), UINT8_C(141), UINT8_C(105),
                        UINT8_C(242), UINT8_C(142), UINT8_C( 33), UINT8_C(177) ),
      simde_x_vloadq_u8(UINT8_C( 44), UINT8_C( 17), UINT8_C(247), UINT8_C(149),
                        UINT8_C(116), UINT8_C( 82), UINT8_C(134), UINT8_C( 16),
                        UINT8_C(152), UINT8_C(243), UINT8_C(125), UINT8_C(100),
                        UINT8_C(111), UINT8_C(120), UINT8_C( 85), UINT8_C( 37) ) },
    { simde_x_vloadq_u8(UINT8_C(220), UINT8_C(174), UINT8_C(192), UINT8_C(215),
                        UINT8_C(181), UINT8_C(131), UINT8_C(155), UINT8_C( 27),
                        UINT8_C(108), UINT8_C(  0), UINT8_C(102), UINT8_C(225),
                        UINT8_C( 47), UINT8_C(  7), UINT8_C(163), UINT8_C( 30) ),
      simde_x_vloadq_u8(UINT8_C(243), UINT8_C( 93), UINT8_C(212), UINT8_C(237),
                        UINT8_C(245), UINT8_C(191), UINT8_C(242), UINT8_C(157),
                        UINT8_C(150), UINT8_C(192), UINT8_C( 58), UINT8_C(113),
                        UINT8_C(167), UINT8_C( 48), UINT8_C(185), UINT8_C(253) ),
      simde_x_vloadq_u8(UINT8_C( 17), UINT8_C(251), UINT8_C(110), UINT8_C( 34),
                        UINT8_C( 20), UINT8_C(215), UINT8_C(109), UINT8_C(208),
                        UINT8_C(141), UINT8_C( 49), UINT8_C( 95), UINT8_C(181),
                        UINT8_C(  7), UINT8_C(255), UINT8_C(197), UINT8_C( 82) ),
      simde_x_vloadq_u8(UINT8_C(250), UINT8_C( 76), UINT8_C( 90), UINT8_C( 12),
                        UINT8_C(212), UINT8_C(155), UINT8_C( 22), UINT8_C( 78),
                        UINT8_C( 99), UINT8_C(113), UINT8_C(139), UINT8_C( 37),
                        UINT8_C(143), UINT8_C(214), UINT8_C(175), UINT8_C(115) ) },
    { simde_x_vloadq_u8(UINT8_C( 91), UINT8_C( 35), UINT8_C(238), UINT8_C(225),
                        UINT8_C(218), UINT8_C( 99), UINT8_C( 58), UINT8_C(220),
                        UINT8_C(165), UINT8_C(144), UINT8_C(168), UINT8_C( 10),
                        UINT8_C( 99), UINT8_C(  0), UINT8_C(250), UINT8_C(182) ),
      simde_x_vloadq_u8(UINT8_C( 12), UINT8_C( 56), UINT8_C( 19), UINT8_C( 34),
                        UINT8_C( 38), UINT8_C( 45), UINT8_C(250), UINT8_C( 90),
                        UINT8_C( 71), UINT8_C(136), UINT8_C(200), UINT8_C( 79),
                        UINT8_C(146), UINT8_C( 57), UINT8_C(204), UINT8_C( 87) ),
      simde_x_vloadq_u8(UINT8_C( 10), UINT8_C(203), UINT8_C(127), UINT8_C( 61),
                        UINT8_C(141), UINT8_C(228), UINT8_C(124), UINT8_C( 45),
                        UINT8_C(118), UINT8_C( 53), UINT8_C(214), UINT8_C(134),
                        UINT8_C(120), UINT8_C(177), UINT8_C(249), UINT8_C(152) ),
      simde_x_vloadq_u8(UINT8_C( 89), UINT8_C(182), UINT8_C( 90), UINT8_C(252),
                        UINT8_C( 65), UINT8_C( 26), UINT8_C(188), UINT8_C(175),
                        UINT8_C(212), UINT8_C( 61), UINT8_C(182), UINT8_C( 65),
                        UINT8_C( 73), UINT8_C(120), UINT8_C( 39), UINT8_C(247) ) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_uint8x16_t a, b, c, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */
  /*   munit_rand_memory(sizeof(c), (uint8_t*) &c); */

  /*   r = simde_vabaq_u8(a, b, c); */

  /*   printf("    { simde_x_vloadq_u8(UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 ") ),\n", */
  /*          a.u8[0], a.u8[1], a.u8[ 2], a.u8[ 3], a.u8[ 4], a.u8[ 5], a.u8[ 6], a.u8[ 7], */
  /*          a.u8[8], a.u8[9], a.u8[10], a.u8[11], a.u8[12], a.u8[13], a.u8[14], a.u8[15]); */
  /*   printf("      simde_x_vloadq_u8(UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 ") ),\n", */
  /*          b.u8[0], b.u8[1], b.u8[ 2], b.u8[ 3], b.u8[ 4], b.u8[ 5], b.u8[ 6], b.u8[ 7], */
  /*          b.u8[8], b.u8[9], b.u8[10], b.u8[11], b.u8[12], b.u8[13], b.u8[14], b.u8[15]); */
  /*   printf("      simde_x_vloadq_u8(UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 ") ),\n", */
  /*          c.u8[0], c.u8[1], c.u8[ 2], c.u8[ 3], c.u8[ 4], c.u8[ 5], c.u8[ 6], c.u8[ 7], */
  /*          c.u8[8], c.u8[9], c.u8[10], c.u8[11], c.u8[12], c.u8[13], c.u8[14], c.u8[15]); */
  /*   printf("      simde_x_vloadq_u8(UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 ") ) },\n", */
  /*          r.u8[0], r.u8[1], r.u8[ 2], r.u8[ 3], r.u8[ 4], r.u8[ 5], r.u8[ 6], r.u8[ 7], */
  /*          r.u8[8], r.u8[9], r.u8[10], r.u8[11], r.u8[12], r.u8[13], r.u8[14], r.u8[15]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint8x16_t r = simde_vabaq_u8(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    simde_neon_assert_uint8x16(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabaq_u16(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_uint16x8_t a;
    simde_uint16x8_t b;
    simde_uint16x8_t c;
    simde_uint16x8_t r;
  } test_vec[8] = {
    { simde_x_vloadq_u16(UINT16_C(53542), UINT16_C(57259), UINT16_C(22339), UINT16_C(12715),
                         UINT16_C(42872), UINT16_C(37334), UINT16_C(53927), UINT16_C(44623)),
      simde_x_vloadq_u16(UINT16_C(37185), UINT16_C(28367), UINT16_C(52012), UINT16_C( 7782),
                         UINT16_C(22217), UINT16_C( 4033), UINT16_C(13479), UINT16_C(17229)),
      simde_x_vloadq_u16(UINT16_C(53798), UINT16_C(10199), UINT16_C( 6843), UINT16_C(  142),
                         UINT16_C(25379), UINT16_C(64669), UINT16_C(23809), UINT16_C(41512)),
      simde_x_vloadq_u16(UINT16_C( 4619), UINT16_C(39091), UINT16_C(42706), UINT16_C( 5075),
                         UINT16_C(46034), UINT16_C(32434), UINT16_C(64257), UINT16_C( 3370)) },
    { simde_x_vloadq_u16(UINT16_C( 4818), UINT16_C(43879), UINT16_C(53895), UINT16_C(34932),
                         UINT16_C(58386), UINT16_C(51971), UINT16_C(46830), UINT16_C(44331)),
      simde_x_vloadq_u16(UINT16_C(17659), UINT16_C(10340), UINT16_C( 6458), UINT16_C(33928),
                         UINT16_C(43316), UINT16_C(44719), UINT16_C(51823), UINT16_C(43333)),
      simde_x_vloadq_u16(UINT16_C(31630), UINT16_C(55632), UINT16_C(39907), UINT16_C(50392),
                         UINT16_C( 6165), UINT16_C(24062), UINT16_C( 4142), UINT16_C( 5980)),
      simde_x_vloadq_u16(UINT16_C(18789), UINT16_C(23635), UINT16_C(21808), UINT16_C(51396),
                         UINT16_C(21235), UINT16_C(31314), UINT16_C(64685), UINT16_C( 6978)) },
    { simde_x_vloadq_u16(UINT16_C(24735), UINT16_C( 2400), UINT16_C(27653), UINT16_C(17049),
                         UINT16_C(22251), UINT16_C(54387), UINT16_C(58984), UINT16_C(42034)),
      simde_x_vloadq_u16(UINT16_C(55851), UINT16_C(   54), UINT16_C( 5910), UINT16_C(63580),
                         UINT16_C(15429), UINT16_C(50516), UINT16_C(30956), UINT16_C( 6870)),
      simde_x_vloadq_u16(UINT16_C(13012), UINT16_C( 3742), UINT16_C(54296), UINT16_C(39629),
                         UINT16_C(13626), UINT16_C( 6485), UINT16_C(25089), UINT16_C(47775)),
      simde_x_vloadq_u16(UINT16_C(47432), UINT16_C( 6088), UINT16_C(10503), UINT16_C(58634),
                         UINT16_C(20448), UINT16_C(10356), UINT16_C(53117), UINT16_C(17403)) },
    { simde_x_vloadq_u16(UINT16_C(38111), UINT16_C(62820), UINT16_C(44334), UINT16_C( 6797),
                         UINT16_C( 2427), UINT16_C( 5948), UINT16_C(44340), UINT16_C( 2613)),
      simde_x_vloadq_u16(UINT16_C( 5569), UINT16_C(26888), UINT16_C( 1736), UINT16_C( 5668),
                         UINT16_C(31418), UINT16_C(53439), UINT16_C(34735), UINT16_C(17073)),
      simde_x_vloadq_u16(UINT16_C(34974), UINT16_C(61606), UINT16_C(46368), UINT16_C(56165),
                         UINT16_C(28056), UINT16_C(26996), UINT16_C(35828), UINT16_C(16558)),
      simde_x_vloadq_u16(UINT16_C( 1980), UINT16_C(32002), UINT16_C(23430), UINT16_C(57294),
                         UINT16_C(64601), UINT16_C(45041), UINT16_C(45433), UINT16_C( 2098)) },
    { simde_x_vloadq_u16(UINT16_C(20537), UINT16_C( 4975), UINT16_C(13127), UINT16_C(48451),
                         UINT16_C(44575), UINT16_C(52378), UINT16_C(59261), UINT16_C(42397)),
      simde_x_vloadq_u16(UINT16_C( 6746), UINT16_C(51277), UINT16_C(47765), UINT16_C(13023),
                         UINT16_C(48671), UINT16_C(40145), UINT16_C(30046), UINT16_C(25854)),
      simde_x_vloadq_u16(UINT16_C(39510), UINT16_C(19320), UINT16_C(29706), UINT16_C(62556),
                         UINT16_C(23474), UINT16_C(33307), UINT16_C(33260), UINT16_C( 1784)),
      simde_x_vloadq_u16(UINT16_C(53301), UINT16_C(38554), UINT16_C(60604), UINT16_C(32448),
                         UINT16_C(19378), UINT16_C(45540), UINT16_C(62475), UINT16_C(18327)) },
    { simde_x_vloadq_u16(UINT16_C(26442), UINT16_C(14473), UINT16_C( 2521), UINT16_C( 4061),
                         UINT16_C( 5123), UINT16_C(42275), UINT16_C(30713), UINT16_C(29733)),
      simde_x_vloadq_u16(UINT16_C( 2610), UINT16_C( 1609), UINT16_C(21193), UINT16_C(17269),
                         UINT16_C(49118), UINT16_C(43571), UINT16_C(36220), UINT16_C(  241)),
      simde_x_vloadq_u16(UINT16_C(46100), UINT16_C(25527), UINT16_C(39780), UINT16_C(17438),
                         UINT16_C(40563), UINT16_C(27021), UINT16_C(36594), UINT16_C(45345)),
      simde_x_vloadq_u16(UINT16_C( 4396), UINT16_C(38391), UINT16_C(21108), UINT16_C( 4230),
                         UINT16_C(62104), UINT16_C(25725), UINT16_C(31087), UINT16_C( 9301)) },
    { simde_x_vloadq_u16(UINT16_C(44764), UINT16_C(55232), UINT16_C(33717), UINT16_C( 7067),
                         UINT16_C(  108), UINT16_C(57702), UINT16_C( 1839), UINT16_C( 7843)),
      simde_x_vloadq_u16(UINT16_C(24051), UINT16_C(60884), UINT16_C(49141), UINT16_C(40434),
                         UINT16_C(49302), UINT16_C(28986), UINT16_C(12455), UINT16_C(64953)),
      simde_x_vloadq_u16(UINT16_C(64273), UINT16_C( 8814), UINT16_C(55060), UINT16_C(53357),
                         UINT16_C(12685), UINT16_C(46431), UINT16_C(65287), UINT16_C(21189)),
      simde_x_vloadq_u16(UINT16_C(19450), UINT16_C( 3162), UINT16_C(39636), UINT16_C(19990),
                         UINT16_C(29027), UINT16_C( 9611), UINT16_C(54671), UINT16_C(29615)) },
    { simde_x_vloadq_u16(UINT16_C( 9051), UINT16_C(57838), UINT16_C(25562), UINT16_C(56378),
                         UINT16_C(37029), UINT16_C( 2728), UINT16_C(   99), UINT16_C(46842)),
      simde_x_vloadq_u16(UINT16_C(14348), UINT16_C( 8723), UINT16_C(11558), UINT16_C(23290),
                         UINT16_C(34887), UINT16_C(20424), UINT16_C(14738), UINT16_C(22476)),
      simde_x_vloadq_u16(UINT16_C(51978), UINT16_C(15743), UINT16_C(58509), UINT16_C(11644),
                         UINT16_C(13686), UINT16_C(34518), UINT16_C(45432), UINT16_C(39161)),
      simde_x_vloadq_u16(UINT16_C(46681), UINT16_C(64858), UINT16_C( 6977), UINT16_C(44732),
                         UINT16_C(15828), UINT16_C(16822), UINT16_C(30793), UINT16_C(63527)) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_uint16x8_t a, b, c, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */
  /*   munit_rand_memory(sizeof(c), (uint8_t*) &c); */

  /*   r = simde_vabaq_u16(a, b, c); */

  /*   printf("    { simde_x_vloadq_u16(UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "),\n" */
  /*          "                         UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 ")),\n", */
  /*          a.u16[0], a.u16[1], a.u16[2], a.u16[3], a.u16[4], a.u16[5], a.u16[6], a.u16[7]); */
  /*   printf("      simde_x_vloadq_u16(UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "),\n" */
  /*          "                         UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 ")),\n", */
  /*          b.u16[0], b.u16[1], b.u16[2], b.u16[3], b.u16[4], b.u16[5], b.u16[6], b.u16[7]); */
  /*   printf("      simde_x_vloadq_u16(UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "),\n" */
  /*          "                         UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 ")),\n", */
  /*          c.u16[0], c.u16[1], c.u16[2], c.u16[3], c.u16[4], c.u16[5], c.u16[6], c.u16[7]); */
  /*   printf("      simde_x_vloadq_u16(UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "),\n" */
  /*          "                         UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 ")) },\n", */
  /*          r.u16[0], r.u16[1], r.u16[2], r.u16[3], r.u16[4], r.u16[5], r.u16[6], r.u16[7]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint16x8_t r = simde_vabaq_u16(test_vec[i].a, test_vec[i].b, test_vec[i].b);
    simde_neon_assert_uint16x8(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabaq_u32(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_uint32x4_t a;
    simde_uint32x4_t b;
    simde_uint32x4_t c;
    simde_uint32x4_t r;
  } test_vec[8] = {
    { simde_x_vloadq_u32(UINT32_C(3752579366), UINT32_C( 833312579),
                         UINT32_C(2446763896), UINT32_C(2924466855)),
      simde_x_vloadq_u32(UINT32_C(1859096897), UINT32_C( 510053164),
                         UINT32_C( 264328905), UINT32_C(1129133223)),
      simde_x_vloadq_u32(UINT32_C( 668455462), UINT32_C(   9312955),
                         UINT32_C(4238172963), UINT32_C(2720554241)),
      simde_x_vloadq_u32(UINT32_C(2561937931), UINT32_C( 332572370),
                         UINT32_C(2125640658), UINT32_C( 220920577)) },
    { simde_x_vloadq_u32(UINT32_C(2875658962), UINT32_C(2289357447),
                         UINT32_C(3406029842), UINT32_C(2905323246)),
      simde_x_vloadq_u32(UINT32_C( 677659899), UINT32_C(2223511866),
                         UINT32_C(2930747700), UINT32_C(2839923311)),
      simde_x_vloadq_u32(UINT32_C(3645930382), UINT32_C(3302530019),
                         UINT32_C(1576933397), UINT32_C( 391909422)),
      simde_x_vloadq_u32(UINT32_C(1548962149), UINT32_C(3368375600),
                         UINT32_C(2052215539), UINT32_C( 457309357)) },
    { simde_x_vloadq_u32(UINT32_C( 157311135), UINT32_C(1117350917),
                         UINT32_C(3564328683), UINT32_C(2754799208)),
      simde_x_vloadq_u32(UINT32_C(   3594795), UINT32_C(4166784790),
                         UINT32_C(3310632005), UINT32_C( 450263276)),
      simde_x_vloadq_u32(UINT32_C( 245248724), UINT32_C(2597180440),
                         UINT32_C( 425014586), UINT32_C(3131007489)),
      simde_x_vloadq_u32(UINT32_C( 398965064), UINT32_C(3842713863),
                         UINT32_C( 678711264), UINT32_C(1140576125)) },
    { simde_x_vloadq_u32(UINT32_C(4117009631), UINT32_C( 445492526),
                         UINT32_C( 389810555), UINT32_C( 171289908)),
      simde_x_vloadq_u32(UINT32_C(1762137537), UINT32_C( 371459784),
                         UINT32_C(3502209722), UINT32_C(1118930863)),
      simde_x_vloadq_u32(UINT32_C(4037445790), UINT32_C(3680875808),
                         UINT32_C(1769237912), UINT32_C(1085180916)),
      simde_x_vloadq_u32(UINT32_C(2097350588), UINT32_C(3754908550),
                         UINT32_C(2951806041), UINT32_C( 137539961)) },
    { simde_x_vloadq_u32(UINT32_C( 326062137), UINT32_C(3175297863),
                         UINT32_C(3432689183), UINT32_C(2778589053)),
      simde_x_vloadq_u32(UINT32_C(3360496218), UINT32_C( 853523093),
                         UINT32_C(2630991391), UINT32_C(1694397790)),
      simde_x_vloadq_u32(UINT32_C(1266195030), UINT32_C(4099699722),
                         UINT32_C(2182831026), UINT32_C( 116949484)),
      simde_x_vloadq_u32(UINT32_C(2526728245), UINT32_C(2126507196),
                         UINT32_C(2984528818), UINT32_C(1201140747)) },
    { simde_x_vloadq_u32(UINT32_C( 948528970), UINT32_C( 266144217),
                         UINT32_C(2770539523), UINT32_C(1948612601)),
      simde_x_vloadq_u32(UINT32_C( 105450034), UINT32_C(1131762377),
                         UINT32_C(2855518174), UINT32_C(  15830396)),
      simde_x_vloadq_u32(UINT32_C(1672983572), UINT32_C(1142856548),
                         UINT32_C(1770888819), UINT32_C(2971766514)),
      simde_x_vloadq_u32(UINT32_C(2516062508), UINT32_C( 277238388),
                         UINT32_C(1685910168), UINT32_C( 609581423)) },
    { simde_x_vloadq_u32(UINT32_C(3619729116), UINT32_C( 463176629),
                         UINT32_C(3781558380), UINT32_C( 514000687)),
      simde_x_vloadq_u32(UINT32_C(3990117875), UINT32_C(2649931765),
                         UINT32_C(1899675798), UINT32_C(4256772263)),
      simde_x_vloadq_u32(UINT32_C( 577698577), UINT32_C(3496859412),
                         UINT32_C(3042914701), UINT32_C(1388707591)),
      simde_x_vloadq_u32(UINT32_C( 207309818), UINT32_C(1310104276),
                         UINT32_C( 629829987), UINT32_C(1940903311)) },
    { simde_x_vloadq_u32(UINT32_C(3790480219), UINT32_C(3694814170),
                         UINT32_C( 178819237), UINT32_C(3069837411)),
      simde_x_vloadq_u32(UINT32_C( 571684876), UINT32_C(1526344998),
                         UINT32_C(1338542151), UINT32_C(1473001874)),
      simde_x_vloadq_u32(UINT32_C(1031785226), UINT32_C( 763159693),
                         UINT32_C(2262185334), UINT32_C(2566500728)),
      simde_x_vloadq_u32(UINT32_C(4250580569), UINT32_C(2931628865),
                         UINT32_C(1102462420), UINT32_C(4163336265)) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_uint32x4_t a, b, c, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */
  /*   munit_rand_memory(sizeof(c), (uint8_t*) &c); */

  /*   r = simde_vabaq_u32(a, b, c); */

  /*   printf("    { simde_x_vloadq_u32(UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 "),\n" */
  /*          "                         UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 ")),\n", */
  /*          a.u32[0], a.u32[1], a.u32[2], a.u32[3]); */
  /*   printf("      simde_x_vloadq_u32(UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 "),\n" */
  /*          "                         UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 ")),\n", */
  /*          b.u32[0], b.u32[1], b.u32[2], b.u32[3]); */
  /*   printf("      simde_x_vloadq_u32(UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 "),\n" */
  /*          "                         UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 ")),\n", */
  /*          c.u32[0], c.u32[1], c.u32[2], c.u32[3]); */
  /*   printf("      simde_x_vloadq_u32(UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 "),\n" */
  /*          "                         UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 ")) },\n", */
  /*          r.u32[0], r.u32[1], r.u32[2], r.u32[3]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint32x4_t r = simde_vabaq_u32(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    simde_neon_assert_uint32x4(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

HEDLEY_DIAGNOSTIC_PUSH
HEDLEY_DIAGNOSTIC_DISABLE_CAST_QUAL

static MunitTest test_suite_tests[] = {
  SIMDE_TESTS_NEON_DEFINE_TEST(s8),
  SIMDE_TESTS_NEON_DEFINE_TEST(s16),
  SIMDE_TESTS_NEON_DEFINE_TEST(s32),
  SIMDE_TESTS_NEON_DEFINE_TEST(u8),
  SIMDE_TESTS_NEON_DEFINE_TEST(u16),
  SIMDE_TESTS_NEON_DEFINE_TEST(u32),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, s8),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, s16),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, s32),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, u8),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, u16),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, u32),

  { NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL }
};

HEDLEY_C_DECL MunitSuite* SIMDE_TESTS_GENERATE_SYMBOL(SIMDE_TESTS_CURRENT_NEON_OP)(void) {
  static MunitSuite suite = { (char*) "/v" HEDLEY_STRINGIFY(SIMDE_TESTS_CURRENT_NEON_OP), test_suite_tests, NULL, 1, MUNIT_SUITE_OPTION_NONE };

  return &suite;
}

HEDLEY_DIAGNOSTIC_POP
