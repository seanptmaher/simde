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
    { simde_x_vload_s8(INT8_C( 102), INT8_C(-122), INT8_C(  50), INT8_C( -67),
                       INT8_C(  95), INT8_C( -94), INT8_C(  37), INT8_C( -96)),
      simde_x_vload_s8(INT8_C(  82), INT8_C(-115), INT8_C( 127), INT8_C(-123),
                       INT8_C(  25), INT8_C(  24), INT8_C( -14), INT8_C( -39)),
      simde_x_vload_s8(INT8_C( -83), INT8_C(  96), INT8_C( -98), INT8_C( -92),
                       INT8_C(  53), INT8_C( -48), INT8_C(  42), INT8_C(   1)),
      simde_x_vload_s8(INT8_C(  11), INT8_C(  89), INT8_C(  19), INT8_C( -36),
                       INT8_C( 123), INT8_C( -22), INT8_C(  93), INT8_C( -56)) },
    { simde_x_vload_s8(INT8_C( 120), INT8_C( -77), INT8_C(  58), INT8_C(  22),
                       INT8_C( -62), INT8_C(  21), INT8_C(  27), INT8_C(-117)),
      simde_x_vload_s8(INT8_C(   3), INT8_C(  89), INT8_C( -55), INT8_C(-107),
                       INT8_C(  32), INT8_C(  52), INT8_C(  51), INT8_C(-128)),
      simde_x_vload_s8(INT8_C( -18), INT8_C(  95), INT8_C(-102), INT8_C(  80),
                       INT8_C(-101), INT8_C(-118), INT8_C(  15), INT8_C( -41)),
      simde_x_vload_s8(INT8_C(-115), INT8_C( -71), INT8_C( 105), INT8_C( -47),
                       INT8_C(  71), INT8_C( -65), INT8_C(  63), INT8_C( -30)) },
    { simde_x_vload_s8(INT8_C( -14), INT8_C(-115), INT8_C( -26), INT8_C( 108),
                       INT8_C( -33), INT8_C( -52), INT8_C(-103), INT8_C( 105)),
      simde_x_vload_s8(INT8_C( 115), INT8_C(-103), INT8_C( -27), INT8_C(  54),
                       INT8_C(  16), INT8_C(  93), INT8_C(  -4), INT8_C( -88)),
      simde_x_vload_s8(INT8_C( -50), INT8_C(   2), INT8_C(-125), INT8_C( -23),
                       INT8_C(-105), INT8_C(  56), INT8_C(-103), INT8_C( -57)),
      simde_x_vload_s8(INT8_C(-105), INT8_C( -10), INT8_C(  72), INT8_C( -71),
                       INT8_C(  88), INT8_C( -15), INT8_C(  -4), INT8_C(-120)) },
    { simde_x_vload_s8(INT8_C( -29), INT8_C( 117), INT8_C(  73), INT8_C( -12),
                       INT8_C(-114), INT8_C(  41), INT8_C(   3), INT8_C(  91)),
      simde_x_vload_s8(INT8_C( -31), INT8_C( -50), INT8_C( -23), INT8_C( -38),
                       INT8_C(  -9), INT8_C(  35), INT8_C(-127), INT8_C( -44)),
      simde_x_vload_s8(INT8_C( -38), INT8_C(-113), INT8_C(  87), INT8_C(  93),
                       INT8_C( 104), INT8_C(   8), INT8_C(-125), INT8_C(  -6)),
      simde_x_vload_s8(INT8_C( -22), INT8_C( -76), INT8_C( -73), INT8_C( 119),
                       INT8_C(  -1), INT8_C(  68), INT8_C(   5), INT8_C(-127)) },
    { simde_x_vload_s8(INT8_C(-115), INT8_C( 124), INT8_C(  40), INT8_C(  82),
                       INT8_C(  56), INT8_C(  53), INT8_C( -97), INT8_C(-123)),
      simde_x_vload_s8(INT8_C(  99), INT8_C(  92), INT8_C(   0), INT8_C( -83),
                       INT8_C(   8), INT8_C( 126), INT8_C( 123), INT8_C( 109)),
      simde_x_vload_s8(INT8_C(  -7), INT8_C(  28), INT8_C(  29), INT8_C(-118),
                       INT8_C(  28), INT8_C( 114), INT8_C(  13), INT8_C(  91)),
      simde_x_vload_s8(INT8_C(  -9), INT8_C( -68), INT8_C(  69), INT8_C( 117),
                       INT8_C(  76), INT8_C(  65), INT8_C(  13), INT8_C(-105)) },
    { simde_x_vload_s8(INT8_C(  67), INT8_C( -11), INT8_C(  14), INT8_C(  51),
                       INT8_C(-111), INT8_C(-117), INT8_C(  18), INT8_C(  17)),
      simde_x_vload_s8(INT8_C(  56), INT8_C( -71), INT8_C( 126), INT8_C(  63),
                       INT8_C(  52), INT8_C(  98), INT8_C(  -4), INT8_C( -63)),
      simde_x_vload_s8(INT8_C(  20), INT8_C(  39), INT8_C( 110), INT8_C( -66),
                       INT8_C(  80), INT8_C(  47), INT8_C( -41), INT8_C( -11)),
      simde_x_vload_s8(INT8_C( 103), INT8_C(  99), INT8_C(  30), INT8_C( -76),
                       INT8_C( -83), INT8_C( -66), INT8_C(  55), INT8_C(  69)) },
    { simde_x_vload_s8(INT8_C( -27), INT8_C(  64), INT8_C( -16), INT8_C(  -1),
                       INT8_C( -17), INT8_C(-104), INT8_C(-115), INT8_C(-121)),
      simde_x_vload_s8(INT8_C(   1), INT8_C( -35), INT8_C( -76), INT8_C( -22),
                       INT8_C( 117), INT8_C( -25), INT8_C( -31), INT8_C( -80)),
      simde_x_vload_s8(INT8_C( -94), INT8_C( -41), INT8_C(  85), INT8_C(  -5),
                       INT8_C(  39), INT8_C( -71), INT8_C(  52), INT8_C(  89)),
      simde_x_vload_s8(INT8_C(  68), INT8_C(  70), INT8_C(-111), INT8_C(  16),
                       INT8_C(  61), INT8_C( -58), INT8_C( -32), INT8_C(  48)) },
    { simde_x_vload_s8(INT8_C(  66), INT8_C( -73), INT8_C( -99), INT8_C( -94),
                       INT8_C( -94), INT8_C(  74), INT8_C( 120), INT8_C(  92)),
      simde_x_vload_s8(INT8_C( -94), INT8_C(  18), INT8_C( -20), INT8_C(  95),
                       INT8_C(-110), INT8_C(  88), INT8_C( -64), INT8_C( -65)),
      simde_x_vload_s8(INT8_C( -80), INT8_C(  18), INT8_C(  98), INT8_C(  -7),
                       INT8_C(  68), INT8_C(  74), INT8_C( 100), INT8_C( -12)),
      simde_x_vload_s8(INT8_C(  80), INT8_C( -73), INT8_C(  19), INT8_C(   8),
                       INT8_C(  84), INT8_C(  88), INT8_C(  28), INT8_C(-111)) },
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
    { simde_x_vload_s16(INT16_C(-31130), INT16_C(-17102), INT16_C(-23969), INT16_C(-24539)),
      simde_x_vload_s16(INT16_C(-29358), INT16_C(-31361), INT16_C(  6169), INT16_C( -9742)),
      simde_x_vload_s16(INT16_C( 24749), INT16_C(-23394), INT16_C(-12235), INT16_C(   298)),
      simde_x_vload_s16(INT16_C( 22977), INT16_C( -9135), INT16_C( -5565), INT16_C(-14499)) },
    { simde_x_vload_s16(INT16_C(-19592), INT16_C(  5690), INT16_C(  5570), INT16_C(-29925)),
      simde_x_vload_s16(INT16_C( 22787), INT16_C(-27191), INT16_C( 13344), INT16_C(-32717)),
      simde_x_vload_s16(INT16_C( 24558), INT16_C( 20634), INT16_C(-30053), INT16_C(-10481)),
      simde_x_vload_s16(INT16_C(-17821), INT16_C(-12021), INT16_C(-16569), INT16_C( -7689)) },
    { simde_x_vload_s16(INT16_C(-29198), INT16_C( 27878), INT16_C(-13089), INT16_C( 27033)),
      simde_x_vload_s16(INT16_C(-26253), INT16_C( 14053), INT16_C( 23824), INT16_C(-22276)),
      simde_x_vload_s16(INT16_C(   718), INT16_C( -5757), INT16_C( 14487), INT16_C(-14439)),
      simde_x_vload_s16(INT16_C( -2227), INT16_C(-17848), INT16_C( -3752), INT16_C(-30666)) },
    { simde_x_vload_s16(INT16_C( 30179), INT16_C( -2999), INT16_C( 10638), INT16_C( 23299)),
      simde_x_vload_s16(INT16_C(-12575), INT16_C( -9495), INT16_C(  9207), INT16_C(-11135)),
      simde_x_vload_s16(INT16_C(-28710), INT16_C( 23895), INT16_C(  2152), INT16_C( -1405)),
      simde_x_vload_s16(INT16_C(-19222), INT16_C( 30391), INT16_C( 17693), INT16_C(-32507)) },
    { simde_x_vload_s16(INT16_C( 31885), INT16_C( 21032), INT16_C( 13624), INT16_C(-31329)),
      simde_x_vload_s16(INT16_C( 23651), INT16_C(-21248), INT16_C( 32264), INT16_C( 28027)),
      simde_x_vload_s16(INT16_C(  7417), INT16_C(-30179), INT16_C( 29212), INT16_C( 23309)),
      simde_x_vload_s16(INT16_C(-17417), INT16_C( 29963), INT16_C( 16676), INT16_C(-26611)) },
    { simde_x_vload_s16(INT16_C( -2749), INT16_C( 13070), INT16_C(-29807), INT16_C(  4370)),
      simde_x_vload_s16(INT16_C(-18120), INT16_C( 16254), INT16_C( 25140), INT16_C(-15876)),
      simde_x_vload_s16(INT16_C( 10004), INT16_C(-16786), INT16_C( 12112), INT16_C( -2601)),
      simde_x_vload_s16(INT16_C( 25375), INT16_C(-19426), INT16_C(-16779), INT16_C( 17645)) },
    { simde_x_vload_s16(INT16_C( 16613), INT16_C(   -16), INT16_C(-26385), INT16_C(-30835)),
      simde_x_vload_s16(INT16_C( -8959), INT16_C( -5452), INT16_C( -6283), INT16_C(-20255)),
      simde_x_vload_s16(INT16_C(-10334), INT16_C( -1195), INT16_C(-18137), INT16_C( 22836)),
      simde_x_vload_s16(INT16_C( 17988), INT16_C(  4241), INT16_C(-14531), INT16_C( 12256)) },
    { simde_x_vload_s16(INT16_C(-18622), INT16_C(-23907), INT16_C( 19106), INT16_C( 23672)),
      simde_x_vload_s16(INT16_C(  4770), INT16_C( 24556), INT16_C( 22674), INT16_C(-16448)),
      simde_x_vload_s16(INT16_C(  4784), INT16_C( -1694), INT16_C( 19012), INT16_C( -2972)),
      simde_x_vload_s16(INT16_C(-18608), INT16_C(  2343), INT16_C( 22768), INT16_C(-28388)) },
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
    { simde_x_vload_s32(INT32_C(-1120762266), INT32_C(-1608146337)),
      simde_x_vload_s32(INT32_C(-2055238318), INT32_C( -638445543)),
      simde_x_vload_s32(INT32_C(-1533124435), INT32_C(   19583029)),
      simde_x_vload_s32(INT32_C( -598648383), INT32_C( -950117765)) },
    { simde_x_vload_s32(INT32_C(  372945784), INT32_C(-1961159230)),
      simde_x_vload_s32(INT32_C(-1781966589), INT32_C(-2144127968)),
      simde_x_vload_s32(INT32_C( 1352294382), INT32_C( -686847333)),
      simde_x_vload_s32(INT32_C( 1533652109), INT32_C( -503878595)) },
    { simde_x_vload_s32(INT32_C( 1827048946), INT32_C( 1771687135)),
      simde_x_vload_s32(INT32_C(  921016691), INT32_C(-1459856112)),
      simde_x_vload_s32(INT32_C( -377290034), INT32_C( -946259817)),
      simde_x_vload_s32(INT32_C(-1169611625), INT32_C(-2009683866)) },
    { simde_x_vload_s32(INT32_C( -196512285), INT32_C( 1526933902)),
      simde_x_vload_s32(INT32_C( -622211359), INT32_C( -729734153)),
      simde_x_vload_s32(INT32_C( 1566019546), INT32_C(  -92075928)),
      simde_x_vload_s32(INT32_C( 1910224106), INT32_C(-2130375169)) },
    { simde_x_vload_s32(INT32_C( 1378385037), INT32_C(-2053163720)),
      simde_x_vload_s32(INT32_C(-1392485277), INT32_C( 1836809736)),
      simde_x_vload_s32(INT32_C(-1977803527), INT32_C( 1527607836)),
      simde_x_vload_s32(INT32_C( 1963703287), INT32_C(-1743961820)) },
    { simde_x_vload_s32(INT32_C(  856618307), INT32_C(  286428049)),
      simde_x_vload_s32(INT32_C( 1065269560), INT32_C(-1040424396)),
      simde_x_vload_s32(INT32_C(-1100077292), INT32_C( -170447024)),
      simde_x_vload_s32(INT32_C(-1308728545), INT32_C( 1156405421)) },
    { simde_x_vload_s32(INT32_C(   -1031963), INT32_C(-2020763409)),
      simde_x_vload_s32(INT32_C( -357245695), INT32_C(-1327372427)),
      simde_x_vload_s32(INT32_C(  -78260318), INT32_C( 1496627495)),
      simde_x_vload_s32(INT32_C(  277953414), INT32_C( -549796035)) },
    { simde_x_vload_s32(INT32_C(-1566722238), INT32_C( 1551387298)),
      simde_x_vload_s32(INT32_C( 1609306786), INT32_C(-1077913454)),
      simde_x_vload_s32(INT32_C( -111013200), INT32_C( -194753980)),
      simde_x_vload_s32(INT32_C(  153597748), INT32_C(-1860420524)) },
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
    { simde_x_vload_u8(UINT8_C(102), UINT8_C(134), UINT8_C( 50), UINT8_C(189),
                       UINT8_C( 95), UINT8_C(162), UINT8_C( 37), UINT8_C(160)),
      simde_x_vload_u8(UINT8_C( 82), UINT8_C(141), UINT8_C(127), UINT8_C(133),
                       UINT8_C( 25), UINT8_C( 24), UINT8_C(242), UINT8_C(217)),
      simde_x_vload_u8(UINT8_C(173), UINT8_C( 96), UINT8_C(158), UINT8_C(164),
                       UINT8_C( 53), UINT8_C(208), UINT8_C( 42), UINT8_C(  1)),
      simde_x_vload_u8(UINT8_C(193), UINT8_C( 89), UINT8_C( 81), UINT8_C(220),
                       UINT8_C(123), UINT8_C( 90), UINT8_C( 93), UINT8_C(200)) },
    { simde_x_vload_u8(UINT8_C(120), UINT8_C(179), UINT8_C( 58), UINT8_C( 22),
                       UINT8_C(194), UINT8_C( 21), UINT8_C( 27), UINT8_C(139)),
      simde_x_vload_u8(UINT8_C(  3), UINT8_C( 89), UINT8_C(201), UINT8_C(149),
                       UINT8_C( 32), UINT8_C( 52), UINT8_C( 51), UINT8_C(128)),
      simde_x_vload_u8(UINT8_C(238), UINT8_C( 95), UINT8_C(154), UINT8_C( 80),
                       UINT8_C(155), UINT8_C(138), UINT8_C( 15), UINT8_C(215)),
      simde_x_vload_u8(UINT8_C( 99), UINT8_C(185), UINT8_C( 11), UINT8_C(209),
                       UINT8_C( 61), UINT8_C(107), UINT8_C(247), UINT8_C(226)) },
    { simde_x_vload_u8(UINT8_C(242), UINT8_C(141), UINT8_C(230), UINT8_C(108),
                       UINT8_C(223), UINT8_C(204), UINT8_C(153), UINT8_C(105)),
      simde_x_vload_u8(UINT8_C(115), UINT8_C(153), UINT8_C(229), UINT8_C( 54),
                       UINT8_C( 16), UINT8_C( 93), UINT8_C(252), UINT8_C(168)),
      simde_x_vload_u8(UINT8_C(206), UINT8_C(  2), UINT8_C(131), UINT8_C(233),
                       UINT8_C(151), UINT8_C( 56), UINT8_C(153), UINT8_C(199)),
      simde_x_vload_u8(UINT8_C( 77), UINT8_C(246), UINT8_C(132), UINT8_C( 31),
                       UINT8_C(102), UINT8_C(167), UINT8_C( 54), UINT8_C(136)) },
    { simde_x_vload_u8(UINT8_C(227), UINT8_C(117), UINT8_C( 73), UINT8_C(244),
                       UINT8_C(142), UINT8_C( 41), UINT8_C(  3), UINT8_C( 91)),
      simde_x_vload_u8(UINT8_C(225), UINT8_C(206), UINT8_C(233), UINT8_C(218),
                       UINT8_C(247), UINT8_C( 35), UINT8_C(129), UINT8_C(212)),
      simde_x_vload_u8(UINT8_C(218), UINT8_C(143), UINT8_C( 87), UINT8_C( 93),
                       UINT8_C(104), UINT8_C(  8), UINT8_C(131), UINT8_C(250)),
      simde_x_vload_u8(UINT8_C(220), UINT8_C( 54), UINT8_C(183), UINT8_C(119),
                       UINT8_C(255), UINT8_C( 14), UINT8_C(  5), UINT8_C(129)) },
    { simde_x_vload_u8(UINT8_C(141), UINT8_C(124), UINT8_C( 40), UINT8_C( 82),
                       UINT8_C( 56), UINT8_C( 53), UINT8_C(159), UINT8_C(133)),
      simde_x_vload_u8(UINT8_C( 99), UINT8_C( 92), UINT8_C(  0), UINT8_C(173),
                       UINT8_C(  8), UINT8_C(126), UINT8_C(123), UINT8_C(109)),
      simde_x_vload_u8(UINT8_C(249), UINT8_C( 28), UINT8_C( 29), UINT8_C(138),
                       UINT8_C( 28), UINT8_C(114), UINT8_C( 13), UINT8_C( 91)),
      simde_x_vload_u8(UINT8_C( 35), UINT8_C( 60), UINT8_C( 69), UINT8_C( 47),
                       UINT8_C( 76), UINT8_C( 41), UINT8_C( 49), UINT8_C(115)) },
    { simde_x_vload_u8(UINT8_C( 67), UINT8_C(245), UINT8_C( 14), UINT8_C( 51),
                       UINT8_C(145), UINT8_C(139), UINT8_C( 18), UINT8_C( 17)),
      simde_x_vload_u8(UINT8_C( 56), UINT8_C(185), UINT8_C(126), UINT8_C( 63),
                       UINT8_C( 52), UINT8_C( 98), UINT8_C(252), UINT8_C(193)),
      simde_x_vload_u8(UINT8_C( 20), UINT8_C( 39), UINT8_C(110), UINT8_C(190),
                       UINT8_C( 80), UINT8_C( 47), UINT8_C(215), UINT8_C(245)),
      simde_x_vload_u8(UINT8_C( 31), UINT8_C( 99), UINT8_C(254), UINT8_C(178),
                       UINT8_C(173), UINT8_C( 88), UINT8_C(237), UINT8_C( 69)) },
    { simde_x_vload_u8(UINT8_C(229), UINT8_C( 64), UINT8_C(240), UINT8_C(255),
                       UINT8_C(239), UINT8_C(152), UINT8_C(141), UINT8_C(135)),
      simde_x_vload_u8(UINT8_C(  1), UINT8_C(221), UINT8_C(180), UINT8_C(234),
                       UINT8_C(117), UINT8_C(231), UINT8_C(225), UINT8_C(176)),
      simde_x_vload_u8(UINT8_C(162), UINT8_C(215), UINT8_C( 85), UINT8_C(251),
                       UINT8_C( 39), UINT8_C(185), UINT8_C( 52), UINT8_C( 89)),
      simde_x_vload_u8(UINT8_C(134), UINT8_C( 58), UINT8_C(145), UINT8_C( 16),
                       UINT8_C(161), UINT8_C(106), UINT8_C(224), UINT8_C( 48)) },
    { simde_x_vload_u8(UINT8_C( 66), UINT8_C(183), UINT8_C(157), UINT8_C(162),
                       UINT8_C(162), UINT8_C( 74), UINT8_C(120), UINT8_C( 92)),
      simde_x_vload_u8(UINT8_C(162), UINT8_C( 18), UINT8_C(236), UINT8_C( 95),
                       UINT8_C(146), UINT8_C( 88), UINT8_C(192), UINT8_C(191)),
      simde_x_vload_u8(UINT8_C(176), UINT8_C( 18), UINT8_C( 98), UINT8_C(249),
                       UINT8_C( 68), UINT8_C( 74), UINT8_C(100), UINT8_C(244)),
      simde_x_vload_u8(UINT8_C( 80), UINT8_C(183), UINT8_C( 19), UINT8_C( 60),
                       UINT8_C( 84), UINT8_C( 60), UINT8_C( 28), UINT8_C(145)) },
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
    { simde_x_vload_u16(UINT16_C(34406), UINT16_C(48434), UINT16_C(41567), UINT16_C(40997)),
      simde_x_vload_u16(UINT16_C(36178), UINT16_C(34175), UINT16_C( 6169), UINT16_C(55794)),
      simde_x_vload_u16(UINT16_C(24749), UINT16_C(42142), UINT16_C(53301), UINT16_C(  298)),
      simde_x_vload_u16(UINT16_C(22977), UINT16_C(56401), UINT16_C(23163), UINT16_C(51037)) },
    { simde_x_vload_u16(UINT16_C(45944), UINT16_C( 5690), UINT16_C( 5570), UINT16_C(35611)),
      simde_x_vload_u16(UINT16_C(22787), UINT16_C(38345), UINT16_C(13344), UINT16_C(32819)),
      simde_x_vload_u16(UINT16_C(24558), UINT16_C(20634), UINT16_C(35483), UINT16_C(55055)),
      simde_x_vload_u16(UINT16_C(47715), UINT16_C(53515), UINT16_C(27709), UINT16_C(57847)) },
    { simde_x_vload_u16(UINT16_C(36338), UINT16_C(27878), UINT16_C(52447), UINT16_C(27033)),
      simde_x_vload_u16(UINT16_C(39283), UINT16_C(14053), UINT16_C(23824), UINT16_C(43260)),
      simde_x_vload_u16(UINT16_C(  718), UINT16_C(59779), UINT16_C(14487), UINT16_C(51097)),
      simde_x_vload_u16(UINT16_C(63309), UINT16_C( 8068), UINT16_C(43110), UINT16_C(34870)) },
    { simde_x_vload_u16(UINT16_C(30179), UINT16_C(62537), UINT16_C(10638), UINT16_C(23299)),
      simde_x_vload_u16(UINT16_C(52961), UINT16_C(56041), UINT16_C( 9207), UINT16_C(54401)),
      simde_x_vload_u16(UINT16_C(36826), UINT16_C(23895), UINT16_C( 2152), UINT16_C(64131)),
      simde_x_vload_u16(UINT16_C(14044), UINT16_C(30391), UINT16_C( 3583), UINT16_C(33029)) },
    { simde_x_vload_u16(UINT16_C(31885), UINT16_C(21032), UINT16_C(13624), UINT16_C(34207)),
      simde_x_vload_u16(UINT16_C(23651), UINT16_C(44288), UINT16_C(32264), UINT16_C(28027)),
      simde_x_vload_u16(UINT16_C( 7417), UINT16_C(35357), UINT16_C(29212), UINT16_C(23309)),
      simde_x_vload_u16(UINT16_C(15651), UINT16_C(12101), UINT16_C(10572), UINT16_C(29489)) },
    { simde_x_vload_u16(UINT16_C(62787), UINT16_C(13070), UINT16_C(35729), UINT16_C( 4370)),
      simde_x_vload_u16(UINT16_C(47416), UINT16_C(16254), UINT16_C(25140), UINT16_C(49660)),
      simde_x_vload_u16(UINT16_C(10004), UINT16_C(48750), UINT16_C(12112), UINT16_C(62935)),
      simde_x_vload_u16(UINT16_C(25375), UINT16_C(45566), UINT16_C(22701), UINT16_C(17645)) },
    { simde_x_vload_u16(UINT16_C(16613), UINT16_C(65520), UINT16_C(39151), UINT16_C(34701)),
      simde_x_vload_u16(UINT16_C(56577), UINT16_C(60084), UINT16_C(59253), UINT16_C(45281)),
      simde_x_vload_u16(UINT16_C(55202), UINT16_C(64341), UINT16_C(47399), UINT16_C(22836)),
      simde_x_vload_u16(UINT16_C(15238), UINT16_C( 4241), UINT16_C(27297), UINT16_C(12256)) },
    { simde_x_vload_u16(UINT16_C(46914), UINT16_C(41629), UINT16_C(19106), UINT16_C(23672)),
      simde_x_vload_u16(UINT16_C( 4770), UINT16_C(24556), UINT16_C(22674), UINT16_C(49088)),
      simde_x_vload_u16(UINT16_C( 4784), UINT16_C(63842), UINT16_C(19012), UINT16_C(62564)),
      simde_x_vload_u16(UINT16_C(46928), UINT16_C(15379), UINT16_C(15444), UINT16_C(37148)) },
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
    { simde_x_vload_u32(UINT32_C(3174205030), UINT32_C(2686820959)),
      simde_x_vload_u32(UINT32_C(2239728978), UINT32_C(3656521753)),
      simde_x_vload_u32(UINT32_C(2761842861), UINT32_C(  19583029)),
      simde_x_vload_u32(UINT32_C(3696318913), UINT32_C(3344849531)) },
    { simde_x_vload_u32(UINT32_C( 372945784), UINT32_C(2333808066)),
      simde_x_vload_u32(UINT32_C(2513000707), UINT32_C(2150839328)),
      simde_x_vload_u32(UINT32_C(1352294382), UINT32_C(3608119963)),
      simde_x_vload_u32(UINT32_C(3507206755), UINT32_C(3791088701)) },
    { simde_x_vload_u32(UINT32_C(1827048946), UINT32_C(1771687135)),
      simde_x_vload_u32(UINT32_C( 921016691), UINT32_C(2835111184)),
      simde_x_vload_u32(UINT32_C(3917677262), UINT32_C(3348707479)),
      simde_x_vload_u32(UINT32_C( 528742221), UINT32_C(2285283430)) },
    { simde_x_vload_u32(UINT32_C(4098455011), UINT32_C(1526933902)),
      simde_x_vload_u32(UINT32_C(3672755937), UINT32_C(3565233143)),
      simde_x_vload_u32(UINT32_C(1566019546), UINT32_C(4202891368)),
      simde_x_vload_u32(UINT32_C(1991718620), UINT32_C(2164592127)) },
    { simde_x_vload_u32(UINT32_C(1378385037), UINT32_C(2241803576)),
      simde_x_vload_u32(UINT32_C(2902482019), UINT32_C(1836809736)),
      simde_x_vload_u32(UINT32_C(2317163769), UINT32_C(1527607836)),
      simde_x_vload_u32(UINT32_C( 793066787), UINT32_C(1932601676)) },
    { simde_x_vload_u32(UINT32_C( 856618307), UINT32_C( 286428049)),
      simde_x_vload_u32(UINT32_C(1065269560), UINT32_C(3254542900)),
      simde_x_vload_u32(UINT32_C(3194890004), UINT32_C(4124520272)),
      simde_x_vload_u32(UINT32_C(2986238751), UINT32_C(1156405421)) },
    { simde_x_vload_u32(UINT32_C(4293935333), UINT32_C(2274203887)),
      simde_x_vload_u32(UINT32_C(3937721601), UINT32_C(2967594869)),
      simde_x_vload_u32(UINT32_C(4216706978), UINT32_C(1496627495)),
      simde_x_vload_u32(UINT32_C( 277953414), UINT32_C( 803236513)) },
    { simde_x_vload_u32(UINT32_C(2728245058), UINT32_C(1551387298)),
      simde_x_vload_u32(UINT32_C(1609306786), UINT32_C(3217053842)),
      simde_x_vload_u32(UINT32_C(4183954096), UINT32_C(4100213316)),
      simde_x_vload_u32(UINT32_C(1007925072), UINT32_C(2434546772)) },
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
    { simde_x_vloadq_s8(INT8_C( 102), INT8_C(-122), INT8_C(  50), INT8_C( -67),
                        INT8_C(  95), INT8_C( -94), INT8_C(  37), INT8_C( -96),
                        INT8_C(  82), INT8_C(-115), INT8_C( 127), INT8_C(-123),
                        INT8_C(  25), INT8_C(  24), INT8_C( -14), INT8_C( -39)),
      simde_x_vloadq_s8(INT8_C( -83), INT8_C(  96), INT8_C( -98), INT8_C( -92),
                        INT8_C(  53), INT8_C( -48), INT8_C(  42), INT8_C(   1),
                        INT8_C( 120), INT8_C( -77), INT8_C(  58), INT8_C(  22),
                        INT8_C( -62), INT8_C(  21), INT8_C(  27), INT8_C(-117)),
      simde_x_vloadq_s8(INT8_C(   3), INT8_C(  89), INT8_C( -55), INT8_C(-107),
                        INT8_C(  32), INT8_C(  52), INT8_C(  51), INT8_C(-128),
                        INT8_C( -18), INT8_C(  95), INT8_C(-102), INT8_C(  80),
                        INT8_C(-101), INT8_C(-118), INT8_C(  15), INT8_C( -41)),
      simde_x_vloadq_s8(INT8_C( -68), INT8_C(-115), INT8_C(  93), INT8_C( -52),
                        INT8_C( 116), INT8_C(   6), INT8_C(  46), INT8_C(  33),
                        INT8_C( -36), INT8_C(  57), INT8_C(  31), INT8_C( -65),
                        INT8_C(  64), INT8_C( -93), INT8_C(  -2), INT8_C(  37)) },
    { simde_x_vloadq_s8(INT8_C( -14), INT8_C(-115), INT8_C( -26), INT8_C( 108),
                        INT8_C( -33), INT8_C( -52), INT8_C(-103), INT8_C( 105),
                        INT8_C( 115), INT8_C(-103), INT8_C( -27), INT8_C(  54),
                        INT8_C(  16), INT8_C(  93), INT8_C(  -4), INT8_C( -88)),
      simde_x_vloadq_s8(INT8_C( -50), INT8_C(   2), INT8_C(-125), INT8_C( -23),
                        INT8_C(-105), INT8_C(  56), INT8_C(-103), INT8_C( -57),
                        INT8_C( -29), INT8_C( 117), INT8_C(  73), INT8_C( -12),
                        INT8_C(-114), INT8_C(  41), INT8_C(   3), INT8_C(  91)),
      simde_x_vloadq_s8(INT8_C( -31), INT8_C( -50), INT8_C( -23), INT8_C( -38),
                        INT8_C(  -9), INT8_C(  35), INT8_C(-127), INT8_C( -44),
                        INT8_C( -38), INT8_C(-113), INT8_C(  87), INT8_C(  93),
                        INT8_C( 104), INT8_C(   8), INT8_C(-125), INT8_C(  -6)),
      simde_x_vloadq_s8(INT8_C(   5), INT8_C( -63), INT8_C(  76), INT8_C( 123),
                        INT8_C(  63), INT8_C( -31), INT8_C( -79), INT8_C( 118),
                        INT8_C( 124), INT8_C( 127), INT8_C( -13), INT8_C( -97),
                        INT8_C( -22), INT8_C( 126), INT8_C( 124), INT8_C(   9)) },
    { simde_x_vloadq_s8(INT8_C(-115), INT8_C( 124), INT8_C(  40), INT8_C(  82),
                        INT8_C(  56), INT8_C(  53), INT8_C( -97), INT8_C(-123),
                        INT8_C(  99), INT8_C(  92), INT8_C(   0), INT8_C( -83),
                        INT8_C(   8), INT8_C( 126), INT8_C( 123), INT8_C( 109)),
      simde_x_vloadq_s8(INT8_C(  -7), INT8_C(  28), INT8_C(  29), INT8_C(-118),
                        INT8_C(  28), INT8_C( 114), INT8_C(  13), INT8_C(  91),
                        INT8_C(  67), INT8_C( -11), INT8_C(  14), INT8_C(  51),
                        INT8_C(-111), INT8_C(-117), INT8_C(  18), INT8_C(  17)),
      simde_x_vloadq_s8(INT8_C(  56), INT8_C( -71), INT8_C( 126), INT8_C(  63),
                        INT8_C(  52), INT8_C(  98), INT8_C(  -4), INT8_C( -63),
                        INT8_C(  20), INT8_C(  39), INT8_C( 110), INT8_C( -66),
                        INT8_C(  80), INT8_C(  47), INT8_C( -41), INT8_C( -11)),
      simde_x_vloadq_s8(INT8_C( -52), INT8_C( -33), INT8_C(-119), INT8_C(   7),
                        INT8_C(  80), INT8_C(  69), INT8_C( -80), INT8_C(  31),
                        INT8_C(-110), INT8_C(-114), INT8_C(  96), INT8_C(  34),
                        INT8_C( -57), INT8_C(  34), INT8_C( -74), INT8_C(-119)) },
    { simde_x_vloadq_s8(INT8_C( -27), INT8_C(  64), INT8_C( -16), INT8_C(  -1),
                        INT8_C( -17), INT8_C(-104), INT8_C(-115), INT8_C(-121),
                        INT8_C(   1), INT8_C( -35), INT8_C( -76), INT8_C( -22),
                        INT8_C( 117), INT8_C( -25), INT8_C( -31), INT8_C( -80)),
      simde_x_vloadq_s8(INT8_C( -94), INT8_C( -41), INT8_C(  85), INT8_C(  -5),
                        INT8_C(  39), INT8_C( -71), INT8_C(  52), INT8_C(  89),
                        INT8_C(  66), INT8_C( -73), INT8_C( -99), INT8_C( -94),
                        INT8_C( -94), INT8_C(  74), INT8_C( 120), INT8_C(  92)),
      simde_x_vloadq_s8(INT8_C( -94), INT8_C(  18), INT8_C( -20), INT8_C(  95),
                        INT8_C(-110), INT8_C(  88), INT8_C( -64), INT8_C( -65),
                        INT8_C( -80), INT8_C(  18), INT8_C(  98), INT8_C(  -7),
                        INT8_C(  68), INT8_C(  74), INT8_C( 100), INT8_C( -12)),
      simde_x_vloadq_s8(INT8_C( -27), INT8_C( 123), INT8_C(  89), INT8_C(  99),
                        INT8_C(-124), INT8_C(  55), INT8_C(   1), INT8_C(  33),
                        INT8_C(-109), INT8_C(  56), INT8_C( 121), INT8_C(  65),
                        INT8_C(  23), INT8_C( -25), INT8_C( -11), INT8_C(  24)) },
    { simde_x_vloadq_s8(INT8_C(  61), INT8_C( -66), INT8_C( -88), INT8_C(  92),
                        INT8_C(-118), INT8_C(-101), INT8_C(  63), INT8_C(  13),
                        INT8_C( -55), INT8_C( -45), INT8_C( 107), INT8_C(  34),
                        INT8_C(  72), INT8_C(  93), INT8_C( -75), INT8_C( 122)),
      simde_x_vloadq_s8(INT8_C(-108), INT8_C( -18), INT8_C(  81), INT8_C( 104),
                        INT8_C(   3), INT8_C( 123), INT8_C(  59), INT8_C( 126),
                        INT8_C(  81), INT8_C(  60), INT8_C(   2), INT8_C( -92),
                        INT8_C( -16), INT8_C(-114), INT8_C(  90), INT8_C(  73)),
      simde_x_vloadq_s8(INT8_C( -91), INT8_C( -22), INT8_C( -28), INT8_C( -10),
                        INT8_C(  67), INT8_C( -74), INT8_C(  70), INT8_C( -70),
                        INT8_C( 107), INT8_C(  26), INT8_C( -31), INT8_C( 109),
                        INT8_C(  32), INT8_C(  -5), INT8_C(   8), INT8_C( 106)),
      simde_x_vloadq_s8(INT8_C(  78), INT8_C( -62), INT8_C(  21), INT8_C( -50),
                        INT8_C( -54), INT8_C(  96), INT8_C(  74), INT8_C( -47),
                        INT8_C( -29), INT8_C( -11), INT8_C(-116), INT8_C( -21),
                        INT8_C( 120), INT8_C( -54), INT8_C(   7), INT8_C(-101)) },
    { simde_x_vloadq_s8(INT8_C(  20), INT8_C(  87), INT8_C( -68), INT8_C( -49),
                        INT8_C(  45), INT8_C(  49), INT8_C(  42), INT8_C( -86),
                        INT8_C( -78), INT8_C(  77), INT8_C( -37), INT8_C(  88),
                        INT8_C(-100), INT8_C( -77), INT8_C(  -9), INT8_C( 115)),
      simde_x_vloadq_s8(INT8_C( -59), INT8_C( -86), INT8_C( -42), INT8_C(  63),
                        INT8_C(  28), INT8_C( -87), INT8_C(  93), INT8_C( -73),
                        INT8_C(  34), INT8_C(  87), INT8_C( -58), INT8_C( 119),
                        INT8_C(   1), INT8_C( -50), INT8_C( -18), INT8_C(   4)),
      simde_x_vloadq_s8(INT8_C( 100), INT8_C( -86), INT8_C( -46), INT8_C( -37),
                        INT8_C( -29), INT8_C( -89), INT8_C(-107), INT8_C( -60),
                        INT8_C(  99), INT8_C( -74), INT8_C( 124), INT8_C( -67),
                        INT8_C( -90), INT8_C(-102), INT8_C( -89), INT8_C( -20)),
      simde_x_vloadq_s8(INT8_C( -77), INT8_C(  87), INT8_C( -64), INT8_C(  51),
                        INT8_C( 102), INT8_C(  51), INT8_C( -14), INT8_C( -73),
                        INT8_C( -13), INT8_C( -18), INT8_C(-111), INT8_C(  18),
                        INT8_C(  -9), INT8_C( -25), INT8_C(  62), INT8_C(-117)) },
    { simde_x_vloadq_s8(INT8_C(   7), INT8_C(  78), INT8_C(  51), INT8_C(  -7),
                        INT8_C(  45), INT8_C( -93), INT8_C(  44), INT8_C(  92),
                        INT8_C(  -4), INT8_C(  28), INT8_C( -69), INT8_C(  84),
                        INT8_C( -62), INT8_C( -14), INT8_C( 110), INT8_C(  90)),
      simde_x_vloadq_s8(INT8_C( -70), INT8_C( -81), INT8_C( -57), INT8_C(  46),
                        INT8_C( -36), INT8_C(  85), INT8_C( -89), INT8_C( -65),
                        INT8_C( -58), INT8_C(  22), INT8_C( -43), INT8_C(  69),
                        INT8_C( -19), INT8_C(-114), INT8_C(-105), INT8_C( -91)),
      simde_x_vloadq_s8(INT8_C(  72), INT8_C(  79), INT8_C(   5), INT8_C( -50),
                        INT8_C( -18), INT8_C(-112), INT8_C(-106), INT8_C( -75),
                        INT8_C( 103), INT8_C(  48), INT8_C(  98), INT8_C(-112),
                        INT8_C(  98), INT8_C(  85), INT8_C(  58), INT8_C(  86)),
      simde_x_vloadq_s8(INT8_C(-107), INT8_C( -18), INT8_C( 113), INT8_C(  89),
                        INT8_C(  63), INT8_C( 104), INT8_C(  61), INT8_C( 102),
                        INT8_C( -99), INT8_C(  54), INT8_C(  72), INT8_C(   9),
                        INT8_C(  55), INT8_C( -71), INT8_C(  17), INT8_C(  11)) },
    { simde_x_vloadq_s8(INT8_C( 100), INT8_C(  65), INT8_C( 104), INT8_C(-110),
                        INT8_C(  43), INT8_C( -32), INT8_C(  99), INT8_C(-121),
                        INT8_C(  54), INT8_C( -64), INT8_C(  43), INT8_C(  64),
                        INT8_C( -68), INT8_C(-116), INT8_C(  48), INT8_C(  19)),
      simde_x_vloadq_s8(INT8_C( 104), INT8_C(-103), INT8_C( -62), INT8_C(-109),
                        INT8_C(  62), INT8_C( 117), INT8_C(  60), INT8_C(  97),
                        INT8_C( -44), INT8_C(  -1), INT8_C( -76), INT8_C( -81),
                        INT8_C(  66), INT8_C(-116), INT8_C( -44), INT8_C(  -4)),
      simde_x_vloadq_s8(INT8_C( 127), INT8_C( -48), INT8_C( -35), INT8_C(  46),
                        INT8_C(-104), INT8_C(  28), INT8_C(  94), INT8_C(  65),
                        INT8_C(  63), INT8_C( -50), INT8_C(   2), INT8_C(  80),
                        INT8_C( -96), INT8_C( -96), INT8_C(  66), INT8_C( 110)),
      simde_x_vloadq_s8(INT8_C( 123), INT8_C( 120), INT8_C(-125), INT8_C(  45),
                        INT8_C( -47), INT8_C(  57), INT8_C(-123), INT8_C( -89),
                        INT8_C( -95), INT8_C( -15), INT8_C( 121), INT8_C( -31),
                        INT8_C(  94), INT8_C( -96), INT8_C( -98), INT8_C(-123)) },
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
    { simde_x_vloadq_s16(INT16_C(-31130), INT16_C(-17102), INT16_C(-23969), INT16_C(-24539),
                         INT16_C(-29358), INT16_C(-31361), INT16_C(  6169), INT16_C( -9742)),
      simde_x_vloadq_s16(INT16_C( 24749), INT16_C(-23394), INT16_C(-12235), INT16_C(   298),
                         INT16_C(-19592), INT16_C(  5690), INT16_C(  5570), INT16_C(-29925)),
      simde_x_vloadq_s16(INT16_C( 22787), INT16_C(-27191), INT16_C( 13344), INT16_C(-32717),
                         INT16_C( 24558), INT16_C( 20634), INT16_C(-30053), INT16_C(-10481)),
      simde_x_vloadq_s16(INT16_C(-29168), INT16_C(-13305), INT16_C(  1610), INT16_C(  8476),
                         INT16_C( 14792), INT16_C(-16417), INT16_C(-23744), INT16_C(  9702)) },
    { simde_x_vloadq_s16(INT16_C(-29198), INT16_C( 27878), INT16_C(-13089), INT16_C( 27033),
                         INT16_C(-26253), INT16_C( 14053), INT16_C( 23824), INT16_C(-22276)),
      simde_x_vloadq_s16(INT16_C(   718), INT16_C( -5757), INT16_C( 14487), INT16_C(-14439),
                         INT16_C( 30179), INT16_C( -2999), INT16_C( 10638), INT16_C( 23299)),
      simde_x_vloadq_s16(INT16_C(-12575), INT16_C( -9495), INT16_C(  9207), INT16_C(-11135),
                         INT16_C(-28710), INT16_C( 23895), INT16_C(  2152), INT16_C( -1405)),
      simde_x_vloadq_s16(INT16_C(-15905), INT16_C( 31616), INT16_C( -7809), INT16_C( 30337),
                         INT16_C( 32636), INT16_C(-24589), INT16_C( 32310), INT16_C(  2428)) },
    { simde_x_vloadq_s16(INT16_C( 31885), INT16_C( 21032), INT16_C( 13624), INT16_C(-31329),
                         INT16_C( 23651), INT16_C(-21248), INT16_C( 32264), INT16_C( 28027)),
      simde_x_vloadq_s16(INT16_C(  7417), INT16_C(-30179), INT16_C( 29212), INT16_C( 23309),
                         INT16_C( -2749), INT16_C( 13070), INT16_C(-29807), INT16_C(  4370)),
      simde_x_vloadq_s16(INT16_C(-18120), INT16_C( 16254), INT16_C( 25140), INT16_C(-15876),
                         INT16_C( 10004), INT16_C(-16786), INT16_C( 12112), INT16_C( -2601)),
      simde_x_vloadq_s16(INT16_C( -8114), INT16_C(  1929), INT16_C( 17696), INT16_C(  7856),
                         INT16_C(-29132), INT16_C(  8608), INT16_C(  8647), INT16_C(-30538)) },
    { simde_x_vloadq_s16(INT16_C( 16613), INT16_C(   -16), INT16_C(-26385), INT16_C(-30835),
                         INT16_C( -8959), INT16_C( -5452), INT16_C( -6283), INT16_C(-20255)),
      simde_x_vloadq_s16(INT16_C(-10334), INT16_C( -1195), INT16_C(-18137), INT16_C( 22836),
                         INT16_C(-18622), INT16_C(-23907), INT16_C( 19106), INT16_C( 23672)),
      simde_x_vloadq_s16(INT16_C(  4770), INT16_C( 24556), INT16_C( 22674), INT16_C(-16448),
                         INT16_C(  4784), INT16_C( -1694), INT16_C( 19012), INT16_C( -2972)),
      simde_x_vloadq_s16(INT16_C( 31717), INT16_C( 25735), INT16_C( 14426), INT16_C(  8449),
                         INT16_C( 14447), INT16_C( 16761), INT16_C( -6189), INT16_C(  6389)) },
    { simde_x_vloadq_s16(INT16_C(-16835), INT16_C( 23720), INT16_C(-25718), INT16_C(  3391),
                         INT16_C(-11319), INT16_C(  8811), INT16_C( 23880), INT16_C( 31413)),
      simde_x_vloadq_s16(INT16_C( -4460), INT16_C( 26705), INT16_C( 31491), INT16_C( 32315),
                         INT16_C( 15441), INT16_C(-23550), INT16_C(-28944), INT16_C( 18778)),
      simde_x_vloadq_s16(INT16_C( -5467), INT16_C( -2332), INT16_C(-18877), INT16_C(-17850),
                         INT16_C(  6763), INT16_C( 28129), INT16_C( -1248), INT16_C( 27144)),
      simde_x_vloadq_s16(INT16_C(-15828), INT16_C(-12779), INT16_C( 24650), INT16_C(-11980),
                         INT16_C( -2641), INT16_C( -5046), INT16_C(-13960), INT16_C(-25757)) },
    { simde_x_vloadq_s16(INT16_C( 22292), INT16_C(-12356), INT16_C( 12589), INT16_C(-21974),
                         INT16_C( 19890), INT16_C( 22747), INT16_C(-19556), INT16_C( 29687)),
      simde_x_vloadq_s16(INT16_C(-21819), INT16_C( 16342), INT16_C(-22244), INT16_C(-18595),
                         INT16_C( 22306), INT16_C( 30662), INT16_C(-12799), INT16_C(  1262)),
      simde_x_vloadq_s16(INT16_C(-21916), INT16_C( -9262), INT16_C(-22557), INT16_C(-15211),
                         INT16_C(-18845), INT16_C(-17028), INT16_C(-25946), INT16_C( -4953)),
      simde_x_vloadq_s16(INT16_C( 22389), INT16_C( 13248), INT16_C( 12902), INT16_C(-18590),
                         INT16_C( -4495), INT16_C(  4901), INT16_C( -6409), INT16_C(-29634)) },
    { simde_x_vloadq_s16(INT16_C( 19975), INT16_C( -1741), INT16_C(-23763), INT16_C( 23596),
                         INT16_C(  7420), INT16_C( 21691), INT16_C( -3390), INT16_C( 23150)),
      simde_x_vloadq_s16(INT16_C(-20550), INT16_C( 11975), INT16_C( 21980), INT16_C(-16473),
                         INT16_C(  5830), INT16_C( 17877), INT16_C(-28947), INT16_C(-23145)),
      simde_x_vloadq_s16(INT16_C( 20296), INT16_C(-12795), INT16_C(-28434), INT16_C(-19050),
                         INT16_C( 12391), INT16_C(-28574), INT16_C( 21858), INT16_C( 22074)),
      simde_x_vloadq_s16(INT16_C( -4715), INT16_C( 23029), INT16_C( 26651), INT16_C( 26173),
                         INT16_C( 13981), INT16_C(  2606), INT16_C(-18121), INT16_C(  2833)) },
    { simde_x_vloadq_s16(INT16_C( 16740), INT16_C(-28056), INT16_C( -8149), INT16_C(-30877),
                         INT16_C(-16330), INT16_C( 16427), INT16_C(-29508), INT16_C(  4912)),
      simde_x_vloadq_s16(INT16_C(-26264), INT16_C(-27710), INT16_C( 30014), INT16_C( 24892),
                         INT16_C(   -44), INT16_C(-20556), INT16_C(-29630), INT16_C(  -812)),
      simde_x_vloadq_s16(INT16_C(-12161), INT16_C( 11997), INT16_C(  7320), INT16_C( 16734),
                         INT16_C(-12737), INT16_C( 20482), INT16_C(-24416), INT16_C( 28226)),
      simde_x_vloadq_s16(INT16_C( 30843), INT16_C( 11651), INT16_C( 14545), INT16_C(-22719),
                         INT16_C( -3637), INT16_C( -8071), INT16_C(-24294), INT16_C(-31586)) },
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
    { simde_x_vloadq_s32(INT32_C(-1120762266), INT32_C(-1608146337),
                         INT32_C(-2055238318), INT32_C( -638445543)),
      simde_x_vloadq_s32(INT32_C(-1533124435), INT32_C(   19583029),
                         INT32_C(  372945784), INT32_C(-1961159230)),
      simde_x_vloadq_s32(INT32_C(-1781966589), INT32_C(-2144127968),
                         INT32_C( 1352294382), INT32_C( -686847333)),
      simde_x_vloadq_s32(INT32_C( -871920112), INT32_C(  523109962),
                         INT32_C(-1075889720), INT32_C(  635866354)) },
    { simde_x_vloadq_s32(INT32_C( 1827048946), INT32_C( 1771687135),
                         INT32_C(  921016691), INT32_C(-1459856112)),
      simde_x_vloadq_s32(INT32_C( -377290034), INT32_C( -946259817),
                         INT32_C( -196512285), INT32_C( 1526933902)),
      simde_x_vloadq_s32(INT32_C( -622211359), INT32_C( -729734153),
                         INT32_C( 1566019546), INT32_C(  -92075928)),
      simde_x_vloadq_s32(INT32_C( 2071970271), INT32_C( 1988212799),
                         INT32_C(-1611418774), INT32_C(  159153718)) },
    { simde_x_vloadq_s32(INT32_C( 1378385037), INT32_C(-2053163720),
                         INT32_C(-1392485277), INT32_C( 1836809736)),
      simde_x_vloadq_s32(INT32_C(-1977803527), INT32_C( 1527607836),
                         INT32_C(  856618307), INT32_C(  286428049)),
      simde_x_vloadq_s32(INT32_C( 1065269560), INT32_C(-1040424396),
                         INT32_C(-1100077292), INT32_C( -170447024)),
      simde_x_vloadq_s32(INT32_C(-1664688050), INT32_C( -326228656),
                         INT32_C(  564210322), INT32_C(-2001282487)) },
    { simde_x_vloadq_s32(INT32_C(   -1031963), INT32_C(-2020763409),
                         INT32_C( -357245695), INT32_C(-1327372427)),
      simde_x_vloadq_s32(INT32_C(  -78260318), INT32_C( 1496627495),
                         INT32_C(-1566722238), INT32_C( 1551387298)),
      simde_x_vloadq_s32(INT32_C( 1609306786), INT32_C(-1077913454),
                         INT32_C( -111013200), INT32_C( -194753980)),
      simde_x_vloadq_s32(INT32_C( 1686535141), INT32_C( -300337062),
                         INT32_C( 1098463343), INT32_C(  418768851)) },
    { simde_x_vloadq_s32(INT32_C( 1554562621), INT32_C(  222272394),
                         INT32_C(  577491913), INT32_C( 2058706248)),
      simde_x_vloadq_s32(INT32_C( 1750199956), INT32_C( 2117827331),
                         INT32_C(-1543357359), INT32_C( 1230671600)),
      simde_x_vloadq_s32(INT32_C( -152769883), INT32_C(-1169770941),
                         INT32_C( 1843468907), INT32_C( 1778973472)),
      simde_x_vloadq_s32(INT32_C( -837434836), INT32_C( 1229641418),
                         INT32_C( 1485632943), INT32_C(-1687959176)) },
    { simde_x_vloadq_s32(INT32_C( -809740524), INT32_C(-1440075475),
                         INT32_C( 1490767282), INT32_C( 1945613212)),
      simde_x_vloadq_s32(INT32_C( 1071033029), INT32_C(-1218598628),
                         INT32_C( 2009487138), INT32_C(   82759169)),
      simde_x_vloadq_s32(INT32_C( -606950812), INT32_C( -996825117),
                         INT32_C(-1115900317), INT32_C( -324560218)),
      simde_x_vloadq_s32(INT32_C(  868243317), INT32_C(-1218301964),
                         INT32_C(-1634620173), INT32_C(-1942034697)) },
    { simde_x_vloadq_s32(INT32_C( -114078201), INT32_C( 1546429229),
                         INT32_C( 1421548796), INT32_C( 1517220546)),
      simde_x_vloadq_s32(INT32_C(  784838586), INT32_C(-1079552548),
                         INT32_C( 1171592902), INT32_C(-1516794131)),
      simde_x_vloadq_s32(INT32_C( -838512824), INT32_C(-1248423698),
                         INT32_C(-1872613273), INT32_C( 1446663522)),
      simde_x_vloadq_s32(INT32_C( 1509273209), INT32_C( 1715300379),
                         INT32_C(-1622657379), INT32_C(-1446237107)) },
    { simde_x_vloadq_s32(INT32_C(-1838661276), INT32_C(-2023497685),
                         INT32_C( 1076609078), INT32_C(  321948860)),
      simde_x_vloadq_s32(INT32_C(-1815963288), INT32_C( 1631352126),
                         INT32_C(-1347092524), INT32_C(  -53179326)),
      simde_x_vloadq_s32(INT32_C(  786288767), INT32_C( 1096686744),
                         INT32_C( 1342361151), INT32_C( 1849860256)),
      simde_x_vloadq_s32(INT32_C( -145946035), INT32_C(-1488832303),
                         INT32_C(-1612844597), INT32_C(-2069978854)) },
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
    { simde_x_vloadq_u8(UINT8_C(102), UINT8_C(134), UINT8_C( 50), UINT8_C(189),
                        UINT8_C( 95), UINT8_C(162), UINT8_C( 37), UINT8_C(160),
                        UINT8_C( 82), UINT8_C(141), UINT8_C(127), UINT8_C(133),
                        UINT8_C( 25), UINT8_C( 24), UINT8_C(242), UINT8_C(217) ),
      simde_x_vloadq_u8(UINT8_C(173), UINT8_C( 96), UINT8_C(158), UINT8_C(164),
                        UINT8_C( 53), UINT8_C(208), UINT8_C( 42), UINT8_C(  1),
                        UINT8_C(120), UINT8_C(179), UINT8_C( 58), UINT8_C( 22),
                        UINT8_C(194), UINT8_C( 21), UINT8_C( 27), UINT8_C(139) ),
      simde_x_vloadq_u8(UINT8_C(  3), UINT8_C( 89), UINT8_C(201), UINT8_C(149),
                        UINT8_C( 32), UINT8_C( 52), UINT8_C( 51), UINT8_C(128),
                        UINT8_C(238), UINT8_C( 95), UINT8_C(154), UINT8_C( 80),
                        UINT8_C(155), UINT8_C(138), UINT8_C( 15), UINT8_C(215) ),
      simde_x_vloadq_u8(UINT8_C(188), UINT8_C(127), UINT8_C( 93), UINT8_C(174),
                        UINT8_C( 74), UINT8_C(  6), UINT8_C( 46), UINT8_C( 31),
                        UINT8_C(200), UINT8_C( 57), UINT8_C(223), UINT8_C(191),
                        UINT8_C(242), UINT8_C(141), UINT8_C(230), UINT8_C( 37) ) },
    { simde_x_vloadq_u8(UINT8_C(242), UINT8_C(141), UINT8_C(230), UINT8_C(108),
                        UINT8_C(223), UINT8_C(204), UINT8_C(153), UINT8_C(105),
                        UINT8_C(115), UINT8_C(153), UINT8_C(229), UINT8_C( 54),
                        UINT8_C( 16), UINT8_C( 93), UINT8_C(252), UINT8_C(168) ),
      simde_x_vloadq_u8(UINT8_C(206), UINT8_C(  2), UINT8_C(131), UINT8_C(233),
                        UINT8_C(151), UINT8_C( 56), UINT8_C(153), UINT8_C(199),
                        UINT8_C(227), UINT8_C(117), UINT8_C( 73), UINT8_C(244),
                        UINT8_C(142), UINT8_C( 41), UINT8_C(  3), UINT8_C( 91) ),
      simde_x_vloadq_u8(UINT8_C(225), UINT8_C(206), UINT8_C(233), UINT8_C(218),
                        UINT8_C(247), UINT8_C( 35), UINT8_C(129), UINT8_C(212),
                        UINT8_C(218), UINT8_C(143), UINT8_C( 87), UINT8_C( 93),
                        UINT8_C(104), UINT8_C(  8), UINT8_C(131), UINT8_C(250) ),
      simde_x_vloadq_u8(UINT8_C(  5), UINT8_C( 89), UINT8_C( 76), UINT8_C( 93),
                        UINT8_C( 63), UINT8_C(183), UINT8_C(129), UINT8_C(118),
                        UINT8_C(106), UINT8_C(179), UINT8_C(243), UINT8_C(159),
                        UINT8_C(234), UINT8_C( 60), UINT8_C(124), UINT8_C( 71) ) },
    { simde_x_vloadq_u8(UINT8_C(141), UINT8_C(124), UINT8_C( 40), UINT8_C( 82),
                        UINT8_C( 56), UINT8_C( 53), UINT8_C(159), UINT8_C(133),
                        UINT8_C( 99), UINT8_C( 92), UINT8_C(  0), UINT8_C(173),
                        UINT8_C(  8), UINT8_C(126), UINT8_C(123), UINT8_C(109) ),
      simde_x_vloadq_u8(UINT8_C(249), UINT8_C( 28), UINT8_C( 29), UINT8_C(138),
                        UINT8_C( 28), UINT8_C(114), UINT8_C( 13), UINT8_C( 91),
                        UINT8_C( 67), UINT8_C(245), UINT8_C( 14), UINT8_C( 51),
                        UINT8_C(145), UINT8_C(139), UINT8_C( 18), UINT8_C( 17) ),
      simde_x_vloadq_u8(UINT8_C( 56), UINT8_C(185), UINT8_C(126), UINT8_C( 63),
                        UINT8_C( 52), UINT8_C( 98), UINT8_C(252), UINT8_C(193),
                        UINT8_C( 20), UINT8_C( 39), UINT8_C(110), UINT8_C(190),
                        UINT8_C( 80), UINT8_C( 47), UINT8_C(215), UINT8_C(245) ),
      simde_x_vloadq_u8(UINT8_C(204), UINT8_C( 25), UINT8_C(137), UINT8_C(  7),
                        UINT8_C( 80), UINT8_C( 37), UINT8_C(142), UINT8_C(235),
                        UINT8_C( 52), UINT8_C(142), UINT8_C( 96), UINT8_C( 56),
                        UINT8_C(199), UINT8_C( 34), UINT8_C( 64), UINT8_C( 81) ) },
    { simde_x_vloadq_u8(UINT8_C(229), UINT8_C( 64), UINT8_C(240), UINT8_C(255),
                        UINT8_C(239), UINT8_C(152), UINT8_C(141), UINT8_C(135),
                        UINT8_C(  1), UINT8_C(221), UINT8_C(180), UINT8_C(234),
                        UINT8_C(117), UINT8_C(231), UINT8_C(225), UINT8_C(176) ),
      simde_x_vloadq_u8(UINT8_C(162), UINT8_C(215), UINT8_C( 85), UINT8_C(251),
                        UINT8_C( 39), UINT8_C(185), UINT8_C( 52), UINT8_C( 89),
                        UINT8_C( 66), UINT8_C(183), UINT8_C(157), UINT8_C(162),
                        UINT8_C(162), UINT8_C( 74), UINT8_C(120), UINT8_C( 92) ),
      simde_x_vloadq_u8(UINT8_C(162), UINT8_C( 18), UINT8_C(236), UINT8_C( 95),
                        UINT8_C(146), UINT8_C( 88), UINT8_C(192), UINT8_C(191),
                        UINT8_C(176), UINT8_C( 18), UINT8_C( 98), UINT8_C(249),
                        UINT8_C( 68), UINT8_C( 74), UINT8_C(100), UINT8_C(244) ),
      simde_x_vloadq_u8(UINT8_C(229), UINT8_C(123), UINT8_C(135), UINT8_C( 99),
                        UINT8_C( 90), UINT8_C( 55), UINT8_C( 25), UINT8_C(237),
                        UINT8_C(111), UINT8_C( 56), UINT8_C(121), UINT8_C( 65),
                        UINT8_C( 23), UINT8_C(231), UINT8_C(205), UINT8_C( 72) ) },
    { simde_x_vloadq_u8(UINT8_C( 61), UINT8_C(190), UINT8_C(168), UINT8_C( 92),
                        UINT8_C(138), UINT8_C(155), UINT8_C( 63), UINT8_C( 13),
                        UINT8_C(201), UINT8_C(211), UINT8_C(107), UINT8_C( 34),
                        UINT8_C( 72), UINT8_C( 93), UINT8_C(181), UINT8_C(122) ),
      simde_x_vloadq_u8(UINT8_C(148), UINT8_C(238), UINT8_C( 81), UINT8_C(104),
                        UINT8_C(  3), UINT8_C(123), UINT8_C( 59), UINT8_C(126),
                        UINT8_C( 81), UINT8_C( 60), UINT8_C(  2), UINT8_C(164),
                        UINT8_C(240), UINT8_C(142), UINT8_C( 90), UINT8_C( 73) ),
      simde_x_vloadq_u8(UINT8_C(165), UINT8_C(234), UINT8_C(228), UINT8_C(246),
                        UINT8_C( 67), UINT8_C(182), UINT8_C( 70), UINT8_C(186),
                        UINT8_C(107), UINT8_C( 26), UINT8_C(225), UINT8_C(109),
                        UINT8_C( 32), UINT8_C(251), UINT8_C(  8), UINT8_C(106) ),
      simde_x_vloadq_u8(UINT8_C( 78), UINT8_C(186), UINT8_C( 59), UINT8_C(234),
                        UINT8_C(202), UINT8_C(214), UINT8_C( 74), UINT8_C( 73),
                        UINT8_C(227), UINT8_C(177), UINT8_C( 74), UINT8_C(235),
                        UINT8_C(120), UINT8_C(202), UINT8_C( 99), UINT8_C(155) ) },
    { simde_x_vloadq_u8(UINT8_C( 20), UINT8_C( 87), UINT8_C(188), UINT8_C(207),
                        UINT8_C( 45), UINT8_C( 49), UINT8_C( 42), UINT8_C(170),
                        UINT8_C(178), UINT8_C( 77), UINT8_C(219), UINT8_C( 88),
                        UINT8_C(156), UINT8_C(179), UINT8_C(247), UINT8_C(115) ),
      simde_x_vloadq_u8(UINT8_C(197), UINT8_C(170), UINT8_C(214), UINT8_C( 63),
                        UINT8_C( 28), UINT8_C(169), UINT8_C( 93), UINT8_C(183),
                        UINT8_C( 34), UINT8_C( 87), UINT8_C(198), UINT8_C(119),
                        UINT8_C(  1), UINT8_C(206), UINT8_C(238), UINT8_C(  4) ),
      simde_x_vloadq_u8(UINT8_C(100), UINT8_C(170), UINT8_C(210), UINT8_C(219),
                        UINT8_C(227), UINT8_C(167), UINT8_C(149), UINT8_C(196),
                        UINT8_C( 99), UINT8_C(182), UINT8_C(124), UINT8_C(189),
                        UINT8_C(166), UINT8_C(154), UINT8_C(167), UINT8_C(236) ),
      simde_x_vloadq_u8(UINT8_C(179), UINT8_C( 87), UINT8_C(184), UINT8_C(107),
                        UINT8_C(244), UINT8_C( 47), UINT8_C( 98), UINT8_C(183),
                        UINT8_C(243), UINT8_C(172), UINT8_C(145), UINT8_C(158),
                        UINT8_C( 65), UINT8_C(127), UINT8_C(176), UINT8_C( 91) ) },
    { simde_x_vloadq_u8(UINT8_C(  7), UINT8_C( 78), UINT8_C( 51), UINT8_C(249),
                        UINT8_C( 45), UINT8_C(163), UINT8_C( 44), UINT8_C( 92),
                        UINT8_C(252), UINT8_C( 28), UINT8_C(187), UINT8_C( 84),
                        UINT8_C(194), UINT8_C(242), UINT8_C(110), UINT8_C( 90) ),
      simde_x_vloadq_u8(UINT8_C(186), UINT8_C(175), UINT8_C(199), UINT8_C( 46),
                        UINT8_C(220), UINT8_C( 85), UINT8_C(167), UINT8_C(191),
                        UINT8_C(198), UINT8_C( 22), UINT8_C(213), UINT8_C( 69),
                        UINT8_C(237), UINT8_C(142), UINT8_C(151), UINT8_C(165) ),
      simde_x_vloadq_u8(UINT8_C( 72), UINT8_C( 79), UINT8_C(  5), UINT8_C(206),
                        UINT8_C(238), UINT8_C(144), UINT8_C(150), UINT8_C(181),
                        UINT8_C(103), UINT8_C( 48), UINT8_C( 98), UINT8_C(144),
                        UINT8_C( 98), UINT8_C( 85), UINT8_C( 58), UINT8_C( 86) ),
      simde_x_vloadq_u8(UINT8_C(149), UINT8_C(238), UINT8_C(113), UINT8_C(153),
                        UINT8_C( 63), UINT8_C(222), UINT8_C( 27), UINT8_C( 82),
                        UINT8_C(157), UINT8_C( 54), UINT8_C( 72), UINT8_C(159),
                        UINT8_C( 55), UINT8_C(185), UINT8_C( 17), UINT8_C( 11) ) },
    { simde_x_vloadq_u8(UINT8_C(100), UINT8_C( 65), UINT8_C(104), UINT8_C(146),
                        UINT8_C( 43), UINT8_C(224), UINT8_C( 99), UINT8_C(135),
                        UINT8_C( 54), UINT8_C(192), UINT8_C( 43), UINT8_C( 64),
                        UINT8_C(188), UINT8_C(140), UINT8_C( 48), UINT8_C( 19) ),
      simde_x_vloadq_u8(UINT8_C(104), UINT8_C(153), UINT8_C(194), UINT8_C(147),
                        UINT8_C( 62), UINT8_C(117), UINT8_C( 60), UINT8_C( 97),
                        UINT8_C(212), UINT8_C(255), UINT8_C(180), UINT8_C(175),
                        UINT8_C( 66), UINT8_C(140), UINT8_C(212), UINT8_C(252) ),
      simde_x_vloadq_u8(UINT8_C(127), UINT8_C(208), UINT8_C(221), UINT8_C( 46),
                        UINT8_C(152), UINT8_C( 28), UINT8_C( 94), UINT8_C( 65),
                        UINT8_C( 63), UINT8_C(206), UINT8_C(  2), UINT8_C( 80),
                        UINT8_C(160), UINT8_C(160), UINT8_C( 66), UINT8_C(110) ),
      simde_x_vloadq_u8(UINT8_C(123), UINT8_C(120), UINT8_C(131), UINT8_C( 45),
                        UINT8_C(133), UINT8_C(135), UINT8_C(133), UINT8_C(103),
                        UINT8_C(161), UINT8_C(143), UINT8_C(121), UINT8_C(225),
                        UINT8_C( 26), UINT8_C(160), UINT8_C(158), UINT8_C(133) ) },
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
    { simde_x_vloadq_u16(UINT16_C(34406), UINT16_C(48434), UINT16_C(41567), UINT16_C(40997),
                         UINT16_C(36178), UINT16_C(34175), UINT16_C( 6169), UINT16_C(55794)),
      simde_x_vloadq_u16(UINT16_C(24749), UINT16_C(42142), UINT16_C(53301), UINT16_C(  298),
                         UINT16_C(45944), UINT16_C( 5690), UINT16_C( 5570), UINT16_C(35611)),
      simde_x_vloadq_u16(UINT16_C(22787), UINT16_C(38345), UINT16_C(13344), UINT16_C(32819),
                         UINT16_C(24558), UINT16_C(20634), UINT16_C(35483), UINT16_C(55055)),
      simde_x_vloadq_u16(UINT16_C(32444), UINT16_C(44637), UINT16_C( 1610), UINT16_C( 7982),
                         UINT16_C(14792), UINT16_C(49119), UINT16_C(36082), UINT16_C( 9702)) },
    { simde_x_vloadq_u16(UINT16_C(36338), UINT16_C(27878), UINT16_C(52447), UINT16_C(27033),
                         UINT16_C(39283), UINT16_C(14053), UINT16_C(23824), UINT16_C(43260)),
      simde_x_vloadq_u16(UINT16_C(  718), UINT16_C(59779), UINT16_C(14487), UINT16_C(51097),
                         UINT16_C(30179), UINT16_C(62537), UINT16_C(10638), UINT16_C(23299)),
      simde_x_vloadq_u16(UINT16_C(52961), UINT16_C(56041), UINT16_C( 9207), UINT16_C(54401),
                         UINT16_C(36826), UINT16_C(23895), UINT16_C( 2152), UINT16_C(64131)),
      simde_x_vloadq_u16(UINT16_C(23045), UINT16_C(24140), UINT16_C(47167), UINT16_C(30337),
                         UINT16_C(45930), UINT16_C(40947), UINT16_C(15338), UINT16_C(18556)) },
    { simde_x_vloadq_u16(UINT16_C(31885), UINT16_C(21032), UINT16_C(13624), UINT16_C(34207),
                         UINT16_C(23651), UINT16_C(44288), UINT16_C(32264), UINT16_C(28027)),
      simde_x_vloadq_u16(UINT16_C( 7417), UINT16_C(35357), UINT16_C(29212), UINT16_C(23309),
                         UINT16_C(62787), UINT16_C(13070), UINT16_C(35729), UINT16_C( 4370)),
      simde_x_vloadq_u16(UINT16_C(47416), UINT16_C(16254), UINT16_C(25140), UINT16_C(49660),
                         UINT16_C(10004), UINT16_C(48750), UINT16_C(12112), UINT16_C(62935)),
      simde_x_vloadq_u16(UINT16_C( 6348), UINT16_C( 1929), UINT16_C( 9552), UINT16_C(60558),
                         UINT16_C(36404), UINT16_C(14432), UINT16_C( 8647), UINT16_C(21056)) },
    { simde_x_vloadq_u16(UINT16_C(16613), UINT16_C(65520), UINT16_C(39151), UINT16_C(34701),
                         UINT16_C(56577), UINT16_C(60084), UINT16_C(59253), UINT16_C(45281)),
      simde_x_vloadq_u16(UINT16_C(55202), UINT16_C(64341), UINT16_C(47399), UINT16_C(22836),
                         UINT16_C(46914), UINT16_C(41629), UINT16_C(19106), UINT16_C(23672)),
      simde_x_vloadq_u16(UINT16_C( 4770), UINT16_C(24556), UINT16_C(22674), UINT16_C(49088),
                         UINT16_C( 4784), UINT16_C(63842), UINT16_C(19012), UINT16_C(62564)),
      simde_x_vloadq_u16(UINT16_C(31717), UINT16_C(25735), UINT16_C(14426), UINT16_C(60953),
                         UINT16_C(14447), UINT16_C(16761), UINT16_C(59159), UINT16_C(18637)) },
    { simde_x_vloadq_u16(UINT16_C(48701), UINT16_C(23720), UINT16_C(39818), UINT16_C( 3391),
                         UINT16_C(54217), UINT16_C( 8811), UINT16_C(23880), UINT16_C(31413)),
      simde_x_vloadq_u16(UINT16_C(61076), UINT16_C(26705), UINT16_C(31491), UINT16_C(32315),
                         UINT16_C(15441), UINT16_C(41986), UINT16_C(36592), UINT16_C(18778)),
      simde_x_vloadq_u16(UINT16_C(60069), UINT16_C(63204), UINT16_C(46659), UINT16_C(47686),
                         UINT16_C( 6763), UINT16_C(28129), UINT16_C(64288), UINT16_C(27144)),
      simde_x_vloadq_u16(UINT16_C(47694), UINT16_C(60219), UINT16_C(54986), UINT16_C(18762),
                         UINT16_C(45539), UINT16_C(60490), UINT16_C(51576), UINT16_C(39779)) },
    { simde_x_vloadq_u16(UINT16_C(22292), UINT16_C(53180), UINT16_C(12589), UINT16_C(43562),
                         UINT16_C(19890), UINT16_C(22747), UINT16_C(45980), UINT16_C(29687)),
      simde_x_vloadq_u16(UINT16_C(43717), UINT16_C(16342), UINT16_C(43292), UINT16_C(46941),
                         UINT16_C(22306), UINT16_C(30662), UINT16_C(52737), UINT16_C( 1262)),
      simde_x_vloadq_u16(UINT16_C(43620), UINT16_C(56274), UINT16_C(42979), UINT16_C(50325),
                         UINT16_C(46691), UINT16_C(48508), UINT16_C(39590), UINT16_C(60583)),
      simde_x_vloadq_u16(UINT16_C(22195), UINT16_C(27576), UINT16_C(12276), UINT16_C(46946),
                         UINT16_C(44275), UINT16_C(40593), UINT16_C(32833), UINT16_C(23472)) },
    { simde_x_vloadq_u16(UINT16_C(19975), UINT16_C(63795), UINT16_C(41773), UINT16_C(23596),
                         UINT16_C( 7420), UINT16_C(21691), UINT16_C(62146), UINT16_C(23150)),
      simde_x_vloadq_u16(UINT16_C(44986), UINT16_C(11975), UINT16_C(21980), UINT16_C(49063),
                         UINT16_C( 5830), UINT16_C(17877), UINT16_C(36589), UINT16_C(42391)),
      simde_x_vloadq_u16(UINT16_C(20296), UINT16_C(52741), UINT16_C(37102), UINT16_C(46486),
                         UINT16_C(12391), UINT16_C(36962), UINT16_C(21858), UINT16_C(22074)),
      simde_x_vloadq_u16(UINT16_C(60821), UINT16_C(39025), UINT16_C(56895), UINT16_C(21019),
                         UINT16_C(13981), UINT16_C(40776), UINT16_C(47415), UINT16_C( 2833)) },
    { simde_x_vloadq_u16(UINT16_C(16740), UINT16_C(37480), UINT16_C(57387), UINT16_C(34659),
                         UINT16_C(49206), UINT16_C(16427), UINT16_C(36028), UINT16_C( 4912)),
      simde_x_vloadq_u16(UINT16_C(39272), UINT16_C(37826), UINT16_C(30014), UINT16_C(24892),
                         UINT16_C(65492), UINT16_C(44980), UINT16_C(35906), UINT16_C(64724)),
      simde_x_vloadq_u16(UINT16_C(53375), UINT16_C(11997), UINT16_C( 7320), UINT16_C(16734),
                         UINT16_C(52799), UINT16_C(20482), UINT16_C(41120), UINT16_C(28226)),
      simde_x_vloadq_u16(UINT16_C(30843), UINT16_C(11651), UINT16_C(34693), UINT16_C(26501),
                         UINT16_C(36513), UINT16_C(57465), UINT16_C(41242), UINT16_C(33950)) },
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
    { simde_x_vloadq_u32(UINT32_C(3174205030), UINT32_C(2686820959),
                         UINT32_C(2239728978), UINT32_C(3656521753)),
      simde_x_vloadq_u32(UINT32_C(2761842861), UINT32_C(  19583029),
                         UINT32_C( 372945784), UINT32_C(2333808066)),
      simde_x_vloadq_u32(UINT32_C(2513000707), UINT32_C(2150839328),
                         UINT32_C(1352294382), UINT32_C(3608119963)),
      simde_x_vloadq_u32(UINT32_C(2925362876), UINT32_C( 523109962),
                         UINT32_C(3219077576), UINT32_C( 635866354)) },
    { simde_x_vloadq_u32(UINT32_C(1827048946), UINT32_C(1771687135),
                         UINT32_C( 921016691), UINT32_C(2835111184)),
      simde_x_vloadq_u32(UINT32_C(3917677262), UINT32_C(3348707479),
                         UINT32_C(4098455011), UINT32_C(1526933902)),
      simde_x_vloadq_u32(UINT32_C(3672755937), UINT32_C(3565233143),
                         UINT32_C(1566019546), UINT32_C(4202891368)),
      simde_x_vloadq_u32(UINT32_C(1582127621), UINT32_C(1988212799),
                         UINT32_C(2683548522), UINT32_C(1216101354)) },
    { simde_x_vloadq_u32(UINT32_C(1378385037), UINT32_C(2241803576),
                         UINT32_C(2902482019), UINT32_C(1836809736)),
      simde_x_vloadq_u32(UINT32_C(2317163769), UINT32_C(1527607836),
                         UINT32_C( 856618307), UINT32_C( 286428049)),
      simde_x_vloadq_u32(UINT32_C(1065269560), UINT32_C(3254542900),
                         UINT32_C(3194890004), UINT32_C(4124520272)),
      simde_x_vloadq_u32(UINT32_C( 126490828), UINT32_C(3968738640),
                         UINT32_C( 945786420), UINT32_C(1379934663)) },
    { simde_x_vloadq_u32(UINT32_C(4293935333), UINT32_C(2274203887),
                         UINT32_C(3937721601), UINT32_C(2967594869)),
      simde_x_vloadq_u32(UINT32_C(4216706978), UINT32_C(1496627495),
                         UINT32_C(2728245058), UINT32_C(1551387298)),
      simde_x_vloadq_u32(UINT32_C(1609306786), UINT32_C(3217053842),
                         UINT32_C(4183954096), UINT32_C(4100213316)),
      simde_x_vloadq_u32(UINT32_C(1686535141), UINT32_C(3994630234),
                         UINT32_C(1098463343), UINT32_C(1221453591)) },
    { simde_x_vloadq_u32(UINT32_C(1554562621), UINT32_C( 222272394),
                         UINT32_C( 577491913), UINT32_C(2058706248)),
      simde_x_vloadq_u32(UINT32_C(1750199956), UINT32_C(2117827331),
                         UINT32_C(2751609937), UINT32_C(1230671600)),
      simde_x_vloadq_u32(UINT32_C(4142197413), UINT32_C(3125196355),
                         UINT32_C(1843468907), UINT32_C(1778973472)),
      simde_x_vloadq_u32(UINT32_C(3946560078), UINT32_C(1229641418),
                         UINT32_C(3964318179), UINT32_C(2607008120)) },
    { simde_x_vloadq_u32(UINT32_C(3485226772), UINT32_C(2854891821),
                         UINT32_C(1490767282), UINT32_C(1945613212)),
      simde_x_vloadq_u32(UINT32_C(1071033029), UINT32_C(3076368668),
                         UINT32_C(2009487138), UINT32_C(  82759169)),
      simde_x_vloadq_u32(UINT32_C(3688016484), UINT32_C(3298142179),
                         UINT32_C(3179066979), UINT32_C(3970407078)),
      simde_x_vloadq_u32(UINT32_C(1807242931), UINT32_C(3076665332),
                         UINT32_C(2660347123), UINT32_C(1538293825)) },
    { simde_x_vloadq_u32(UINT32_C(4180889095), UINT32_C(1546429229),
                         UINT32_C(1421548796), UINT32_C(1517220546)),
      simde_x_vloadq_u32(UINT32_C( 784838586), UINT32_C(3215414748),
                         UINT32_C(1171592902), UINT32_C(2778173165)),
      simde_x_vloadq_u32(UINT32_C(3456454472), UINT32_C(3046543598),
                         UINT32_C(2422354023), UINT32_C(1446663522)),
      simde_x_vloadq_u32(UINT32_C(2557537685), UINT32_C(1377558079),
                         UINT32_C(2672309917), UINT32_C( 185710903)) },
    { simde_x_vloadq_u32(UINT32_C(2456306020), UINT32_C(2271469611),
                         UINT32_C(1076609078), UINT32_C( 321948860)),
      simde_x_vloadq_u32(UINT32_C(2479004008), UINT32_C(1631352126),
                         UINT32_C(2947874772), UINT32_C(4241787970)),
      simde_x_vloadq_u32(UINT32_C( 786288767), UINT32_C(1096686744),
                         UINT32_C(1342361151), UINT32_C(1849860256)),
      simde_x_vloadq_u32(UINT32_C( 763590779), UINT32_C(1736804229),
                         UINT32_C(3766062753), UINT32_C(2224988442)) },
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
