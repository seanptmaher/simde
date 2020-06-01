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
#define SIMDE_TESTS_CURRENT_NEON_OP abd
#include <test/arm/neon/test-neon-internal.h>
#include <simde/arm/neon.h>

static MunitResult
test_simde_vabd_s8(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int8x8_t a;
    simde_int8x8_t b;
    simde_int8x8_t r;
  } test_vec[8] = {
    { simde_x_vload_s8(INT8_C(  51), INT8_C(  54), INT8_C(  77), INT8_C(  36),
                       INT8_C( -12), INT8_C(-101), INT8_C(-121), INT8_C( -83)),
      simde_x_vload_s8(INT8_C(-107), INT8_C(  37), INT8_C(  43), INT8_C(  51),
                       INT8_C( -43), INT8_C(   7), INT8_C( -20), INT8_C(-108)),
      simde_x_vload_s8(INT8_C( -98), INT8_C(  17), INT8_C(  34), INT8_C(  15),
                       INT8_C(  31), INT8_C( 108), INT8_C( 101), INT8_C(  25)) },
    { simde_x_vload_s8(INT8_C( -90), INT8_C(  50), INT8_C( -11), INT8_C( -74),
                       INT8_C( -14), INT8_C(  32), INT8_C( -58), INT8_C(  62)),
      simde_x_vload_s8(INT8_C(  30), INT8_C(-101), INT8_C(-101), INT8_C(  50),
                       INT8_C(-101), INT8_C(-106), INT8_C( -87), INT8_C( -17)),
      simde_x_vload_s8(INT8_C( 120), INT8_C(-105), INT8_C(  90), INT8_C( 124),
                       INT8_C(  87), INT8_C(-118), INT8_C(  29), INT8_C(  79)) },
    { simde_x_vload_s8(INT8_C(  26), INT8_C( -13), INT8_C(  12), INT8_C( -88),
                       INT8_C( -85), INT8_C(  92), INT8_C(  37), INT8_C( -25)),
      simde_x_vload_s8(INT8_C( -94), INT8_C( -55), INT8_C(  60), INT8_C(  22),
                       INT8_C(  75), INT8_C(  54), INT8_C(  41), INT8_C(  30)),
      simde_x_vload_s8(INT8_C( 120), INT8_C(  42), INT8_C(  48), INT8_C( 110),
                       INT8_C( -96), INT8_C(  38), INT8_C(   4), INT8_C(  55)) },
    { simde_x_vload_s8(INT8_C( 127), INT8_C(-111), INT8_C(-117), INT8_C( -22),
                       INT8_C(   1), INT8_C( -98), INT8_C( 110), INT8_C(  49)),
      simde_x_vload_s8(INT8_C(-108), INT8_C(  51), INT8_C( -54), INT8_C( -49),
                       INT8_C(  95), INT8_C(-118), INT8_C( -26), INT8_C(  74)),
      simde_x_vload_s8(INT8_C( -21), INT8_C( -94), INT8_C(  63), INT8_C(  27),
                       INT8_C(  94), INT8_C(  20), INT8_C(-120), INT8_C(  25)) },
    { simde_x_vload_s8(INT8_C( -90), INT8_C( -36), INT8_C( 116), INT8_C( -51),
                       INT8_C(  23), INT8_C( 109), INT8_C( -52), INT8_C(-126)),
      simde_x_vload_s8(INT8_C( 109), INT8_C( -64), INT8_C(-123), INT8_C( -71),
                       INT8_C( 110), INT8_C( 101), INT8_C(  46), INT8_C(  24)),
      simde_x_vload_s8(INT8_C( -57), INT8_C(  28), INT8_C( -17), INT8_C(  20),
                       INT8_C(  87), INT8_C(   8), INT8_C(  98), INT8_C(-106)) },
    { simde_x_vload_s8(INT8_C(  64), INT8_C( -82), INT8_C(  50), INT8_C(  57),
                       INT8_C( -13), INT8_C(  -3), INT8_C(-125), INT8_C(   1)),
      simde_x_vload_s8(INT8_C(  -9), INT8_C(   2), INT8_C( -10), INT8_C( -53),
                       INT8_C(-114), INT8_C(  32), INT8_C( -37), INT8_C( 102)),
      simde_x_vload_s8(INT8_C(  73), INT8_C(  84), INT8_C(  60), INT8_C( 110),
                       INT8_C( 101), INT8_C(  35), INT8_C(  88), INT8_C( 101)) },
    { simde_x_vload_s8(INT8_C( -48), INT8_C( 108), INT8_C(   6), INT8_C(  71),
                       INT8_C( -14), INT8_C(  11), INT8_C(  81), INT8_C( -88)),
      simde_x_vload_s8(INT8_C( -21), INT8_C( -71), INT8_C(  15), INT8_C( -65),
                       INT8_C(  88), INT8_C(  11), INT8_C( 109), INT8_C( 101)),
      simde_x_vload_s8(INT8_C(  27), INT8_C( -77), INT8_C(   9), INT8_C(-120),
                       INT8_C( 102), INT8_C(   0), INT8_C(  28), INT8_C( -67)) },
    { simde_x_vload_s8(INT8_C(-123), INT8_C(  -3), INT8_C( 121), INT8_C( -44),
                       INT8_C( -34), INT8_C( -73), INT8_C( -51), INT8_C(-119)),
      simde_x_vload_s8(INT8_C( -96), INT8_C(  37), INT8_C(  73), INT8_C(-102),
                       INT8_C( 121), INT8_C( 124), INT8_C(  32), INT8_C( -24)),
      simde_x_vload_s8(INT8_C(  27), INT8_C(  40), INT8_C(  48), INT8_C(  58),
                       INT8_C(-101), INT8_C( -59), INT8_C(  83), INT8_C(  95)) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_int8x8_t a, b, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */

  /*   r = simde_vabd_s8(a, b); */

  /*   printf("    { simde_x_vload_s8(INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                       INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 ")),\n", */
  /*          a.i8[0], a.i8[1], a.i8[2], a.i8[3], a.i8[4], a.i8[5], a.i8[6], a.i8[7]); */
  /*   printf("      simde_x_vload_s8(INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                       INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 ")),\n", */
  /*          b.i8[0], b.i8[1], b.i8[2], b.i8[3], b.i8[4], b.i8[5], b.i8[6], b.i8[7]); */
  /*   printf("      simde_x_vload_s8(INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n" */
  /*          "                       INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 ")) },\n", */
  /*          r.i8[0], r.i8[1], r.i8[2], r.i8[3], r.i8[4], r.i8[5], r.i8[6], r.i8[7]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int8x8_t r = simde_vabd_s8(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_int8x8(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabd_s16(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int16x4_t a;
    simde_int16x4_t b;
    simde_int16x4_t r;
  } test_vec[8] = {
    { simde_x_vload_s16(INT16_C( 13875), INT16_C(  9293), INT16_C(-25612), INT16_C(-21113)),
      simde_x_vload_s16(INT16_C(  9621), INT16_C( 13099), INT16_C(  2005), INT16_C(-27412)),
      simde_x_vload_s16(INT16_C(  4254), INT16_C(  3806), INT16_C( 27617), INT16_C(  6299)) },
    { simde_x_vload_s16(INT16_C( 12966), INT16_C(-18699), INT16_C(  8434), INT16_C( 16070)),
      simde_x_vload_s16(INT16_C(-25826), INT16_C( 12955), INT16_C(-26981), INT16_C( -4183)),
      simde_x_vload_s16(INT16_C(-26744), INT16_C( 31654), INT16_C(-30121), INT16_C( 20253)) },
    { simde_x_vload_s16(INT16_C( -3302), INT16_C(-22516), INT16_C( 23723), INT16_C( -6363)),
      simde_x_vload_s16(INT16_C(-13918), INT16_C(  5692), INT16_C( 13899), INT16_C(  7721)),
      simde_x_vload_s16(INT16_C( 10616), INT16_C( 28208), INT16_C(  9824), INT16_C( 14084)) },
    { simde_x_vload_s16(INT16_C(-28289), INT16_C( -5493), INT16_C(-25087), INT16_C( 12654)),
      simde_x_vload_s16(INT16_C( 13204), INT16_C(-12342), INT16_C(-30113), INT16_C( 19174)),
      simde_x_vload_s16(INT16_C(-24043), INT16_C(  6849), INT16_C(  5026), INT16_C(  6520)) },
    { simde_x_vload_s16(INT16_C( -9050), INT16_C(-12940), INT16_C( 27927), INT16_C(-32052)),
      simde_x_vload_s16(INT16_C(-16275), INT16_C(-18043), INT16_C( 25966), INT16_C(  6190)),
      simde_x_vload_s16(INT16_C(  7225), INT16_C(  5103), INT16_C(  1961), INT16_C(-27294)) },
    { simde_x_vload_s16(INT16_C(-20928), INT16_C( 14642), INT16_C(  -525), INT16_C(   387)),
      simde_x_vload_s16(INT16_C(   759), INT16_C(-13322), INT16_C(  8334), INT16_C( 26331)),
      simde_x_vload_s16(INT16_C( 21687), INT16_C( 27964), INT16_C(  8859), INT16_C( 25944)) },
    { simde_x_vload_s16(INT16_C( 27856), INT16_C( 18182), INT16_C(  3058), INT16_C(-22447)),
      simde_x_vload_s16(INT16_C(-17941), INT16_C(-16625), INT16_C(  2904), INT16_C( 25965)),
      simde_x_vload_s16(INT16_C(-19739), INT16_C(-30729), INT16_C(   154), INT16_C(-17124)) },
    { simde_x_vload_s16(INT16_C(  -635), INT16_C(-11143), INT16_C(-18466), INT16_C(-30259)),
      simde_x_vload_s16(INT16_C(  9632), INT16_C(-26039), INT16_C( 31865), INT16_C( -6112)),
      simde_x_vload_s16(INT16_C( 10267), INT16_C( 14896), INT16_C(-15205), INT16_C( 24147)) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_int16x4_t a, b, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */

  /*   r = simde_vabd_s16(a, b); */

  /*   printf("    { simde_x_vload_s16(INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 ")),\n", */
  /*          a.i16[0], a.i16[1], a.i16[2], a.i16[3]); */
  /*   printf("      simde_x_vload_s16(INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 ")),\n", */
  /*          b.i16[0], b.i16[1], b.i16[2], b.i16[3]); */
  /*   printf("      simde_x_vload_s16(INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 ")) },\n", */
  /*          r.i16[0], r.i16[1], r.i16[2], r.i16[3]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int16x4_t r = simde_vabd_s16(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_int16x4(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabd_s32(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int32x2_t a;
    simde_int32x2_t b;
    simde_int32x2_t r;
  } test_vec[8] = {
    { simde_x_vload_s32(INT32_C(  609039923), INT32_C(-1383621644)),
      simde_x_vload_s32(INT32_C(  858465685), INT32_C(-1796470827)),
      simde_x_vload_s32(INT32_C(  249425762), INT32_C(  412849183)) },
    { simde_x_vload_s32(INT32_C(-1225444698), INT32_C( 1053171954)),
      simde_x_vload_s32(INT32_C(  849058590), INT32_C( -274098533)),
      simde_x_vload_s32(INT32_C( 2074503288), INT32_C( 1327270487)) },
    { simde_x_vload_s32(INT32_C(-1475546342), INT32_C( -416981845)),
      simde_x_vload_s32(INT32_C(  373082530), INT32_C(  506017355)),
      simde_x_vload_s32(INT32_C( 1848628872), INT32_C(  922999200)) },
    { simde_x_vload_s32(INT32_C( -359952001), INT32_C(  829332993)),
      simde_x_vload_s32(INT32_C( -808832108), INT32_C( 1256622687)),
      simde_x_vload_s32(INT32_C(  448880107), INT32_C(  427289694)) },
    { simde_x_vload_s32(INT32_C( -847979354), INT32_C(-2100531945)),
      simde_x_vload_s32(INT32_C(-1182416787), INT32_C(  405693806)),
      simde_x_vload_s32(INT32_C(  334437433), INT32_C(-1788741545)) },
    { simde_x_vload_s32(INT32_C(  959622720), INT32_C(   25427443)),
      simde_x_vload_s32(INT32_C( -873069833), INT32_C( 1725636750)),
      simde_x_vload_s32(INT32_C( 1832692553), INT32_C( 1700209307)) },
    { simde_x_vload_s32(INT32_C( 1191603408), INT32_C(-1471083534)),
      simde_x_vload_s32(INT32_C(-1089488405), INT32_C( 1701645144)),
      simde_x_vload_s32(INT32_C(-2013875483), INT32_C(-1122238618)) },
    { simde_x_vload_s32(INT32_C( -730202747), INT32_C(-1983006754)),
      simde_x_vload_s32(INT32_C(-1706482272), INT32_C( -400524167)),
      simde_x_vload_s32(INT32_C(  976279525), INT32_C( 1582482587)) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_int32x2_t a, b, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */

  /*   r = simde_vabd_s32(a, b); */

  /*   printf("    { simde_x_vload_s32(INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 ")),\n", */
  /*          a.i32[0], a.i32[1]); */
  /*   printf("      simde_x_vload_s32(INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 ")),\n", */
  /*          b.i32[0], b.i32[1]); */
  /*   printf("      simde_x_vload_s32(INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 ")) },\n", */
  /*          r.i32[0], r.i32[1]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int32x2_t r = simde_vabd_s32(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_int32x2(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabd_u8(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_uint8x8_t a;
    simde_uint8x8_t b;
    simde_uint8x8_t r;
  } test_vec[8] = {
    { simde_x_vload_u8(UINT8_C(159), UINT8_C( 38), UINT8_C(231), UINT8_C(196),
                       UINT8_C(182), UINT8_C( 85), UINT8_C( 77), UINT8_C(239)),
      simde_x_vload_u8(UINT8_C( 55), UINT8_C(224), UINT8_C(254), UINT8_C( 34),
                       UINT8_C(135), UINT8_C(235), UINT8_C(246), UINT8_C(157)),
      simde_x_vload_u8(UINT8_C(104), UINT8_C(186), UINT8_C( 23), UINT8_C(162),
                       UINT8_C( 47), UINT8_C(150), UINT8_C(169), UINT8_C( 82)) },
    { simde_x_vload_u8(UINT8_C(120), UINT8_C( 78), UINT8_C( 89), UINT8_C(202),
                       UINT8_C(245), UINT8_C( 10), UINT8_C(105), UINT8_C( 19)),
      simde_x_vload_u8(UINT8_C(111), UINT8_C(  9), UINT8_C( 30), UINT8_C(189),
                       UINT8_C( 89), UINT8_C( 64), UINT8_C(201), UINT8_C(234)),
      simde_x_vload_u8(UINT8_C(  9), UINT8_C( 69), UINT8_C( 59), UINT8_C( 13),
                       UINT8_C(156), UINT8_C( 54), UINT8_C( 96), UINT8_C(215)) },
    { simde_x_vload_u8(UINT8_C(133), UINT8_C(114), UINT8_C(  9), UINT8_C(254),
                       UINT8_C(148), UINT8_C(237), UINT8_C( 48), UINT8_C(158)),
      simde_x_vload_u8(UINT8_C( 70), UINT8_C( 43), UINT8_C(184), UINT8_C(157),
                       UINT8_C(177), UINT8_C( 53), UINT8_C(173), UINT8_C(111)),
      simde_x_vload_u8(UINT8_C( 63), UINT8_C( 71), UINT8_C(175), UINT8_C( 97),
                       UINT8_C( 29), UINT8_C(184), UINT8_C(125), UINT8_C( 47)) },
    { simde_x_vload_u8(UINT8_C( 78), UINT8_C(171), UINT8_C(228), UINT8_C( 69),
                       UINT8_C(139), UINT8_C(197), UINT8_C( 56), UINT8_C(247)),
      simde_x_vload_u8(UINT8_C( 53), UINT8_C( 46), UINT8_C( 27), UINT8_C(134),
                       UINT8_C(145), UINT8_C(111), UINT8_C(180), UINT8_C(112)),
      simde_x_vload_u8(UINT8_C( 25), UINT8_C(125), UINT8_C(201), UINT8_C( 65),
                       UINT8_C(  6), UINT8_C( 86), UINT8_C(124), UINT8_C(135)) },
    { simde_x_vload_u8(UINT8_C(188), UINT8_C(171), UINT8_C( 40), UINT8_C( 29),
                       UINT8_C(168), UINT8_C( 60), UINT8_C( 88), UINT8_C(208)),
      simde_x_vload_u8(UINT8_C( 99), UINT8_C(108), UINT8_C(227), UINT8_C(128),
                       UINT8_C(158), UINT8_C(232), UINT8_C(242), UINT8_C(159)),
      simde_x_vload_u8(UINT8_C( 89), UINT8_C( 63), UINT8_C(187), UINT8_C( 99),
                       UINT8_C( 10), UINT8_C(172), UINT8_C(154), UINT8_C( 49)) },
    { simde_x_vload_u8(UINT8_C( 46), UINT8_C( 18), UINT8_C(131), UINT8_C( 17),
                       UINT8_C( 27), UINT8_C(141), UINT8_C(100), UINT8_C(165)),
      simde_x_vload_u8(UINT8_C(214), UINT8_C(221), UINT8_C(186), UINT8_C( 97),
                       UINT8_C(162), UINT8_C(188), UINT8_C(130), UINT8_C(164)),
      simde_x_vload_u8(UINT8_C(168), UINT8_C(203), UINT8_C( 55), UINT8_C( 80),
                       UINT8_C(135), UINT8_C( 47), UINT8_C( 30), UINT8_C(  1)) },
    { simde_x_vload_u8(UINT8_C(176), UINT8_C(169), UINT8_C( 88), UINT8_C(189),
                       UINT8_C(113), UINT8_C(238), UINT8_C(  4), UINT8_C(216)),
      simde_x_vload_u8(UINT8_C( 69), UINT8_C(152), UINT8_C( 55), UINT8_C( 20),
                       UINT8_C( 19), UINT8_C( 46), UINT8_C(220), UINT8_C(207)),
      simde_x_vload_u8(UINT8_C(107), UINT8_C( 17), UINT8_C( 33), UINT8_C(169),
                       UINT8_C( 94), UINT8_C(192), UINT8_C(216), UINT8_C(  9)) },
    { simde_x_vload_u8(UINT8_C( 59), UINT8_C(155), UINT8_C( 10), UINT8_C(232),
                       UINT8_C(187), UINT8_C( 41), UINT8_C(162), UINT8_C(120)),
      simde_x_vload_u8(UINT8_C(239), UINT8_C(180), UINT8_C( 94), UINT8_C(248),
                       UINT8_C( 59), UINT8_C(128), UINT8_C( 74), UINT8_C( 22)),
      simde_x_vload_u8(UINT8_C(180), UINT8_C( 25), UINT8_C( 84), UINT8_C( 16),
                       UINT8_C(128), UINT8_C( 87), UINT8_C( 88), UINT8_C( 98)) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_uint8x8_t a, b, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */

  /*   r = simde_vabd_u8(a, b); */

  /*   printf("    { simde_x_vload_u8(UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                       UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 ")),\n", */
  /*          a.u8[0], a.u8[1], a.u8[2], a.u8[3], a.u8[4], a.u8[5], a.u8[6], a.u8[7]); */
  /*   printf("      simde_x_vload_u8(UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                       UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 ")),\n", */
  /*          b.u8[0], b.u8[1], b.u8[2], b.u8[3], b.u8[4], b.u8[5], b.u8[6], b.u8[7]); */
  /*   printf("      simde_x_vload_u8(UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n" */
  /*          "                       UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 ")) },\n", */
  /*          r.u8[0], r.u8[1], r.u8[2], r.u8[3], r.u8[4], r.u8[5], r.u8[6], r.u8[7]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint8x8_t r = simde_vabd_u8(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_uint8x8(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabd_u16(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_uint16x4_t a;
    simde_uint16x4_t b;
    simde_uint16x4_t r;
  } test_vec[8] = {
    { simde_x_vload_u16(UINT16_C( 9887), UINT16_C(50407), UINT16_C(21942), UINT16_C(61261)),
      simde_x_vload_u16(UINT16_C(57399), UINT16_C( 8958), UINT16_C(60295), UINT16_C(40438)),
      simde_x_vload_u16(UINT16_C(47512), UINT16_C(41449), UINT16_C(38353), UINT16_C(20823)) },
    { simde_x_vload_u16(UINT16_C(20088), UINT16_C(51801), UINT16_C( 2805), UINT16_C( 4969)),
      simde_x_vload_u16(UINT16_C( 2415), UINT16_C(48414), UINT16_C(16473), UINT16_C(60105)),
      simde_x_vload_u16(UINT16_C(17673), UINT16_C( 3387), UINT16_C(13668), UINT16_C(55136)) },
    { simde_x_vload_u16(UINT16_C(29317), UINT16_C(65033), UINT16_C(60820), UINT16_C(40496)),
      simde_x_vload_u16(UINT16_C(11078), UINT16_C(40376), UINT16_C(13745), UINT16_C(28589)),
      simde_x_vload_u16(UINT16_C(18239), UINT16_C(24657), UINT16_C(47075), UINT16_C(11907)) },
    { simde_x_vload_u16(UINT16_C(43854), UINT16_C(17892), UINT16_C(50571), UINT16_C(63288)),
      simde_x_vload_u16(UINT16_C(11829), UINT16_C(34331), UINT16_C(28561), UINT16_C(28852)),
      simde_x_vload_u16(UINT16_C(32025), UINT16_C(16439), UINT16_C(22010), UINT16_C(34436)) },
    { simde_x_vload_u16(UINT16_C(43964), UINT16_C( 7464), UINT16_C(15528), UINT16_C(53336)),
      simde_x_vload_u16(UINT16_C(27747), UINT16_C(32995), UINT16_C(59550), UINT16_C(40946)),
      simde_x_vload_u16(UINT16_C(16217), UINT16_C(25531), UINT16_C(44022), UINT16_C(12390)) },
    { simde_x_vload_u16(UINT16_C( 4654), UINT16_C( 4483), UINT16_C(36123), UINT16_C(42340)),
      simde_x_vload_u16(UINT16_C(56790), UINT16_C(25018), UINT16_C(48290), UINT16_C(42114)),
      simde_x_vload_u16(UINT16_C(52136), UINT16_C(20535), UINT16_C(12167), UINT16_C(  226)) },
    { simde_x_vload_u16(UINT16_C(43440), UINT16_C(48472), UINT16_C(61041), UINT16_C(55300)),
      simde_x_vload_u16(UINT16_C(38981), UINT16_C( 5175), UINT16_C(11795), UINT16_C(53212)),
      simde_x_vload_u16(UINT16_C( 4459), UINT16_C(43297), UINT16_C(49246), UINT16_C( 2088)) },
    { simde_x_vload_u16(UINT16_C(39739), UINT16_C(59402), UINT16_C(10683), UINT16_C(30882)),
      simde_x_vload_u16(UINT16_C(46319), UINT16_C(63582), UINT16_C(32827), UINT16_C( 5706)),
      simde_x_vload_u16(UINT16_C( 6580), UINT16_C( 4180), UINT16_C(22144), UINT16_C(25176)) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_uint16x4_t a, b, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */

  /*   r = simde_vabd_u16(a, b); */

  /*   printf("    { simde_x_vload_u16(UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 ")),\n", */
  /*          a.u16[0], a.u16[1], a.u16[2], a.u16[3]); */
  /*   printf("      simde_x_vload_u16(UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 ")),\n", */
  /*          b.u16[0], b.u16[1], b.u16[2], b.u16[3]); */
  /*   printf("      simde_x_vload_u16(UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 ")) },\n", */
  /*          r.u16[0], r.u16[1], r.u16[2], r.u16[3]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint16x4_t r = simde_vabd_u16(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_uint16x4(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabd_u32(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_uint32x2_t a;
    simde_uint32x2_t b;
    simde_uint32x2_t r;
  } test_vec[8] = {
    { simde_x_vload_u32(UINT32_C(3303483039), UINT32_C(4014822838)),
      simde_x_vload_u32(UINT32_C( 587128887), UINT32_C(2650205063)),
      simde_x_vload_u32(UINT32_C(2716354152), UINT32_C(1364617775)) },
    { simde_x_vload_u32(UINT32_C(3394850424), UINT32_C( 325651189)),
      simde_x_vload_u32(UINT32_C(3172862319), UINT32_C(3939057753)),
      simde_x_vload_u32(UINT32_C( 221988105), UINT32_C(3613406564)) },
    { simde_x_vload_u32(UINT32_C(4262032005), UINT32_C(2654006676)),
      simde_x_vload_u32(UINT32_C(2646092614), UINT32_C(1873622449)),
      simde_x_vload_u32(UINT32_C(1615939391), UINT32_C( 780384227)) },
    { simde_x_vload_u32(UINT32_C(1172613966), UINT32_C(4147692939)),
      simde_x_vload_u32(UINT32_C(2249928245), UINT32_C(1890873233)),
      simde_x_vload_u32(UINT32_C(1077314279), UINT32_C(2256819706)) },
    { simde_x_vload_u32(UINT32_C( 489204668), UINT32_C(3495443624)),
      simde_x_vload_u32(UINT32_C(2162388067), UINT32_C(2683496606)),
      simde_x_vload_u32(UINT32_C(1673183399), UINT32_C( 811947018)) },
    { simde_x_vload_u32(UINT32_C( 293802542), UINT32_C(2774830363)),
      simde_x_vload_u32(UINT32_C(1639636438), UINT32_C(2760031394)),
      simde_x_vload_u32(UINT32_C(1345833896), UINT32_C(  14798969)) },
    { simde_x_vload_u32(UINT32_C(3176704432), UINT32_C(3624201841)),
      simde_x_vload_u32(UINT32_C( 339187781), UINT32_C(3487313427)),
      simde_x_vload_u32(UINT32_C(2837516651), UINT32_C( 136888414)) },
    { simde_x_vload_u32(UINT32_C(3893009211), UINT32_C(2023893435)),
      simde_x_vload_u32(UINT32_C(4166956271), UINT32_C( 373981243)),
      simde_x_vload_u32(UINT32_C( 273947060), UINT32_C(1649912192)) },

  };

  /* printf("\n"); */
  /* for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) { */
  /*   simde_uint32x2_t a, b, r; */

  /*   munit_rand_memory(sizeof(a), (uint8_t*) &a); */
  /*   munit_rand_memory(sizeof(b), (uint8_t*) &b); */

  /*   r = simde_vabd_u32(a, b); */

  /*   printf("    { simde_x_vload_u32(UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 ")),\n", */
  /*          a.u32[0], a.u32[1]); */
  /*   printf("      simde_x_vload_u32(UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 ")),\n", */
  /*          b.u32[0], b.u32[1]); */
  /*   printf("      simde_x_vload_u32(UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 ")) },\n", */
  /*          r.u32[0], r.u32[1]); */
  /* } */
  /* return MUNIT_FAIL; */

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint32x2_t r = simde_vabd_u32(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_uint32x2(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

#if 0

static MunitResult
test_simde_vabd_f32(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_float32x2_t a;
    simde_float32x2_t b;
    simde_float32x2_t r;
  } test_vec[8] = {

  };

  printf("\n");
  for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) {
    simde_float32x2_t a, b, r;

    a = simde_neon_random_float32x2();
    b = simde_neon_random_float32x2();

    r = simde_vabd_f32(a, b);

    printf("    { simde_x_vload_f32(SIMDE_FLOAT32_C(%*.2f), SIMDE_FLOAT32_C(%*.2f)),\n",
           8, a.f32[0], 8, a.f32[1]);
    printf("      simde_x_vload_f32(SIMDE_FLOAT32_C(%*.2f), SIMDE_FLOAT32_C(%*.2f)),\n",
           8, b.f32[0], 8, b.f32[1]);
    printf("      simde_x_vload_f32(SIMDE_FLOAT32_C(%*.2f), SIMDE_FLOAT32_C(%*.2f)) },\n",
           8, r.f32[0], 8, r.f32[1]);
  }
  return MUNIT_FAIL;

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_float32x2_t r = simde_vabd_f32(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_float32x2_equal(r, test_vec[i].r, 1);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabd_f64(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_float64x1_t a;
    simde_float64x1_t b;
    simde_float64x1_t r;
  } test_vec[8] = {

  };

  printf("\n");
  for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) {
    simde_float64x1_t a, b, r;

    a = simde_neon_random_float64x1();
    b = simde_neon_random_float64x1();

    r = simde_vabd_f64(a, b);

    printf("    { simde_x_vload_f64(SIMDE_FLOAT64_C(%*.2f)),\n", 8, a.f64[0]);
    printf("      simde_x_vload_f64(SIMDE_FLOAT64_C(%*.2f)),\n", 8, b.f64[0]);
    printf("      simde_x_vload_f64(SIMDE_FLOAT64_C(%*.2f)) },\n", 8, r.f64[0]);
  }
  return MUNIT_FAIL;

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_float64x1_t r = simde_vabd_f64(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_float64x1_equal(r, test_vec[i].r, 1);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabdq_s8(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int8x16_t a;
    simde_int8x16_t b;
    simde_int8x16_t r;
  } test_vec[8] = {

  };

  printf("\n");
  for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) {
    simde_int8x16_t a, b, r;

    munit_rand_memory(sizeof(a), (uint8_t*) &a);
    munit_rand_memory(sizeof(b), (uint8_t*) &b);

    r = simde_vabdq_s8(a, b);

    printf("    { simde_x_vloadq_s8(INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n"
	   "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n"
           "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n"
	   "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 ")),\n",
	   a.i8[ 0], a.i8[ 1], a.i8[ 2], a.i8[ 3], a.i8[ 4], a.i8[ 5], a.i8[ 6], a.i8[ 7],
           a.i8[ 8], a.i8[ 9], a.i8[10], a.i8[11], a.i8[12], a.i8[13], a.i8[14], a.i8[15]);
    printf("      simde_x_vloadq_s8(INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n"
	   "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n"
           "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n"
	   "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 ")),\n",
	   b.i8[ 0], b.i8[ 1], b.i8[ 2], b.i8[ 3], b.i8[ 4], b.i8[ 5], b.i8[ 6], b.i8[ 7],
           b.i8[ 8], b.i8[ 9], b.i8[10], b.i8[11], b.i8[12], b.i8[13], b.i8[14], b.i8[15]);
    printf("      simde_x_vloadq_s8(INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n"
	   "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n"
           "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "),\n"
	   "                        INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 "), INT8_C(%4" PRId8 ")) },\n",
	   r.i8[ 0], r.i8[ 1], r.i8[ 2], r.i8[ 3], r.i8[ 4], r.i8[ 5], r.i8[ 6], r.i8[ 7],
           r.i8[ 8], r.i8[ 9], r.i8[10], r.i8[11], r.i8[12], r.i8[13], r.i8[14], r.i8[15]);
  }
  return MUNIT_FAIL;

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int8x16_t r = simde_vabdq_s8(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_int8x16(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabdq_s16(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int16x8_t a;
    simde_int16x8_t b;
    simde_int16x8_t r;
  } test_vec[8] = {

  };

  printf("\n");
  for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) {
    simde_int16x8_t a, b, r;

    munit_rand_memory(sizeof(a), (uint8_t*) &a);
    munit_rand_memory(sizeof(b), (uint8_t*) &b);

    r = simde_vabdq_s16(a, b);

    printf("    { simde_x_vloadq_s16(INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "),\n"
	   "                         INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 ")),\n",
           a.i16[0], a.i16[1], a.i16[2], a.i16[3], a.i16[4], a.i16[5], a.i16[6], a.i16[7]);
    printf("      simde_x_vloadq_s16(INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "),\n"
	   "                         INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 ")),\n",
           b.i16[0], b.i16[1], b.i16[2], b.i16[3], b.i16[4], b.i16[5], b.i16[6], b.i16[7]);
    printf("      simde_x_vloadq_s16(INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "),\n"
	   "                         INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 "), INT16_C(%6" PRId16 ")) },\n",
           r.i16[0], r.i16[1], r.i16[2], r.i16[3], r.i16[4], r.i16[5], r.i16[6], r.i16[7]);
  }
  return MUNIT_FAIL;

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int16x8_t r = simde_vabdq_s16(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_int16x8(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabdq_s32(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_int32x4_t a;
    simde_int32x4_t b;
    simde_int32x4_t r;
  } test_vec[8] = {

  };

  printf("\n");
  for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) {
    simde_int32x4_t a, b, r;

    munit_rand_memory(sizeof(a), (uint8_t*) &a);
    munit_rand_memory(sizeof(b), (uint8_t*) &b);

    r = simde_vabdq_s32(a, b);

    printf("    { simde_x_vloadq_s32(INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 "),\n"
	   "                         INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 ")),\n",
           a.i32[0], a.i32[1], a.i32[2], a.i32[3]);
    printf("      simde_x_vloadq_s32(INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 "),\n"
	   "                         INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 ")),\n",
           b.i32[0], b.i32[1], b.i32[2], b.i32[3]);
    printf("      simde_x_vloadq_s32(INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 "),\n"
	   "                         INT32_C(%11" PRId32 "), INT32_C(%11" PRId32 ")) },\n",
           r.i32[0], r.i32[1], r.i32[2], r.i32[3]);
  }
  return MUNIT_FAIL;

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int32x4_t r = simde_vabdq_s32(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_int32x4(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabdq_s64(const MunitParameter params[], void* data) {
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
    simde_int64x2_t a, b, r;

    munit_rand_memory(sizeof(a), (uint8_t*) &a);
    munit_rand_memory(sizeof(b), (uint8_t*) &b);

    r = simde_vabdq_s64(a, b);

    printf("    { simde_x_vloadq_s64(INT64_C(%21" PRId64 "), INT64_C(%21" PRId64 ")),\n", a.i64[0], a.i64[1]);
    printf("      simde_x_vloadq_s64(INT64_C(%21" PRId64 "), INT64_C(%21" PRId64 ")),\n", b.i64[0], b.i64[1]);
    printf("      simde_x_vloadq_s64(INT64_C(%21" PRId64 "), INT64_C(%21" PRId64 ")) },\n", r.i64[0], r.i64[1]);
  }
  return MUNIT_FAIL;

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int64x2_t r = simde_vabdq_s64(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_int64x2(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabdq_u8(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_uint8x16_t a;
    simde_uint8x16_t b;
    simde_uint8x16_t r;
  } test_vec[8] = {

  };

  printf("\n");
  for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) {
    simde_uint8x16_t a, b, r;

    munit_rand_memory(sizeof(a), (uint8_t*) &a);
    munit_rand_memory(sizeof(b), (uint8_t*) &b);

    r = simde_vabdq_u8(a, b);

    printf("    { simde_x_vloadq_u8(UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n"
           "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n"
           "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n"
           "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 ") ),\n",
           a.u8[0], a.u8[1], a.u8[ 2], a.u8[ 3], a.u8[ 4], a.u8[ 5], a.u8[ 6], a.u8[ 7],
           a.u8[8], a.u8[9], a.u8[10], a.u8[11], a.u8[12], a.u8[13], a.u8[14], a.u8[15]);
    printf("      simde_x_vloadq_u8(UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n"
           "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n"
           "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n"
           "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 ") ),\n",
           b.u8[0], b.u8[1], b.u8[ 2], b.u8[ 3], b.u8[ 4], b.u8[ 5], b.u8[ 6], b.u8[ 7],
           b.u8[8], b.u8[9], b.u8[10], b.u8[11], b.u8[12], b.u8[13], b.u8[14], b.u8[15]);
    printf("      simde_x_vloadq_u8(UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n"
           "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n"
           "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "),\n"
           "                        UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 "), UINT8_C(%3" PRIu8 ") ) },\n",
           r.u8[0], r.u8[1], r.u8[ 2], r.u8[ 3], r.u8[ 4], r.u8[ 5], r.u8[ 6], r.u8[ 7],
           r.u8[8], r.u8[9], r.u8[10], r.u8[11], r.u8[12], r.u8[13], r.u8[14], r.u8[15]);
  }
  return MUNIT_FAIL;

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint8x16_t r = simde_vabdq_u8(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_uint8x16(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabdq_u16(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_uint16x8_t a;
    simde_uint16x8_t b;
    simde_uint16x8_t r;
  } test_vec[8] = {

  };

  printf("\n");
  for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) {
    simde_uint16x8_t a, b, r;

    munit_rand_memory(sizeof(a), (uint8_t*) &a);
    munit_rand_memory(sizeof(b), (uint8_t*) &b);

    r = simde_vabdq_u16(a, b);

    printf("    { simde_x_vloadq_u16(UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "),\n"
           "                         UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 ")),\n",
           a.u16[0], a.u16[1], a.u16[2], a.u16[3], a.u16[4], a.u16[5], a.u16[6], a.u16[7]);
    printf("      simde_x_vloadq_u16(UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "),\n"
           "                         UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 ")),\n",
           b.u16[0], b.u16[1], b.u16[2], b.u16[3], b.u16[4], b.u16[5], b.u16[6], b.u16[7]);
    printf("      simde_x_vloadq_u16(UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "),\n"
           "                         UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 "), UINT16_C(%5" PRIu16 ")) },\n",
           r.u16[0], r.u16[1], r.u16[2], r.u16[3], r.u16[4], r.u16[5], r.u16[6], r.u16[7]);
  }
  return MUNIT_FAIL;

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint16x8_t r = simde_vabdq_u16(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_uint16x8(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabdq_u32(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_uint32x4_t a;
    simde_uint32x4_t b;
    simde_uint32x4_t r;
  } test_vec[8] = {

  };

  printf("\n");
  for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) {
    simde_uint32x4_t a, b, r;

    munit_rand_memory(sizeof(a), (uint8_t*) &a);
    munit_rand_memory(sizeof(b), (uint8_t*) &b);

    r = simde_vabdq_u32(a, b);

    printf("    { simde_x_vloadq_u32(UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 "),\n"
           "                         UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 ")),\n",
           a.u32[0], a.u32[1], a.u32[2], a.u32[3]);
    printf("      simde_x_vloadq_u32(UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 "),\n"
           "                         UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 ")),\n",
           b.u32[0], b.u32[1], b.u32[2], b.u32[3]);
    printf("      simde_x_vloadq_u32(UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 "),\n"
           "                         UINT32_C(%10" PRIu32 "), UINT32_C(%10" PRIu32 ")) },\n",
           r.u32[0], r.u32[1], r.u32[2], r.u32[3]);
  }
  return MUNIT_FAIL;

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint32x4_t r = simde_vabdq_u32(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_uint32x4(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabdq_u64(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_uint64x2_t a;
    simde_uint64x2_t b;
    simde_uint64x2_t r;
  } test_vec[8] = {

  };

  printf("\n");
  for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) {
    simde_uint64x2_t a, b, r;

    munit_rand_memory(sizeof(a), (uint8_t*) &a);
    munit_rand_memory(sizeof(b), (uint8_t*) &b);

    r = simde_vabdq_u64(a, b);

    printf("    { simde_x_vloadq_u64(UINT64_C(%19" PRIu64 "), UINT64_C(%19" PRIu64 ")),\n",
           a.u64[0], a.u64[1]);
    printf("      simde_x_vloadq_u64(UINT64_C(%19" PRIu64 "), UINT64_C(%19" PRIu64 ")),\n",
           b.u64[0], b.u64[1]);
    printf("      simde_x_vloadq_u64(UINT64_C(%19" PRIu64 "), UINT64_C(%19" PRIu64 ")) },\n",
           r.u64[0], r.u64[1]);
  }
  return MUNIT_FAIL;

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint64x2_t r = simde_vabdq_u64(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_uint64x2(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabdq_f32(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_float32x4_t a;
    simde_float32x4_t b;
    simde_float32x4_t r;
  } test_vec[8] = {

  };

  printf("\n");
  for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) {
    simde_float32x4_t a, b, r;

    a = simde_neon_random_float32x4();
    b = simde_neon_random_float32x4();

    r = simde_vabdq_f32(a, b);

    printf("    { simde_x_vloadq_f32(SIMDE_FLOAT32_C(%*.2f), SIMDE_FLOAT32_C(%*.2f),\n"
           "                         SIMDE_FLOAT32_C(%*.2f), SIMDE_FLOAT32_C(%*.2f)),\n",
           8, a.f32[0], 8, a.f32[1], 8, a.f32[2], 8, a.f32[3]);
    printf("      simde_x_vloadq_f32(SIMDE_FLOAT32_C(%*.2f), SIMDE_FLOAT32_C(%*.2f),\n"
           "                         SIMDE_FLOAT32_C(%*.2f), SIMDE_FLOAT32_C(%*.2f)),\n",
           8, b.f32[0], 8, b.f32[1], 8, b.f32[2], 8, b.f32[3]);
    printf("      simde_x_vloadq_f32(SIMDE_FLOAT32_C(%*.2f), SIMDE_FLOAT32_C(%*.2f),\n"
           "                         SIMDE_FLOAT32_C(%*.2f), SIMDE_FLOAT32_C(%*.2f)) },\n",
           8, r.f32[0], 8, r.f32[1], 8, r.f32[2], 8, r.f32[3]);
  }
  return MUNIT_FAIL;

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_float32x4_t r = simde_vabdq_f32(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_float32x4_equal(r, test_vec[i].r, 1);
  }

  return MUNIT_OK;
}

static MunitResult
test_simde_vabdq_f64(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_float64x2_t a;
    simde_float64x2_t b;
    simde_float64x2_t r;
  } test_vec[8] = {

  };

  printf("\n");
  for (size_t i = 0 ; i < (sizeof(test_vec) / (sizeof(test_vec[0]))) ; i++) {
    simde_float64x2_t a, b, r;

    a = simde_neon_random_float64x2();
    b = simde_neon_random_float64x2();

    r = simde_vabdq_f64(a, b);

    printf("    { simde_x_vloadq_f64(SIMDE_FLOAT64_C(%*.2f), SIMDE_FLOAT64_C(%*.2f)),\n",
           8, a.f64[0], 8, a.f64[1]);
    printf("      simde_x_vloadq_f64(SIMDE_FLOAT64_C(%*.2f), SIMDE_FLOAT64_C(%*.2f)),\n",
           8, b.f64[0], 8, b.f64[1]);
    printf("      simde_x_vloadq_f64(SIMDE_FLOAT64_C(%*.2f), SIMDE_FLOAT64_C(%*.2f)) },\n",
           8, r.f64[0], 8, r.f64[1]);
  }
  return MUNIT_FAIL;

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_float64x2_t r = simde_vabdq_f64(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_float64x2_equal(r, test_vec[i].r, 1);
  }

  return MUNIT_OK;
}
#endif

HEDLEY_DIAGNOSTIC_PUSH
HEDLEY_DIAGNOSTIC_DISABLE_CAST_QUAL

static MunitTest vabd_tests[] = {
  SIMDE_TESTS_NEON_DEFINE_TEST(s8),
  SIMDE_TESTS_NEON_DEFINE_TEST(s16),
  SIMDE_TESTS_NEON_DEFINE_TEST(s32),
  // SIMDE_TESTS_NEON_DEFINE_TEST(s64),
  SIMDE_TESTS_NEON_DEFINE_TEST(u8),
  SIMDE_TESTS_NEON_DEFINE_TEST(u16),
  SIMDE_TESTS_NEON_DEFINE_TEST(u32),
  // SIMDE_TESTS_NEON_DEFINE_TEST(u64),
  // SIMDE_TESTS_NEON_DEFINE_TEST(f32),
  // SIMDE_TESTS_NEON_DEFINE_TEST(f64),
  // SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, s8),
  // SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, s16),
  // SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, s32),
  // SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, s64),
  // SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, u8),
  // SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, u16),
  // SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, u32),
  // SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, u64),
  // SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, f32),
  // SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, f64),

  { NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL }
};

HEDLEY_C_DECL MunitSuite* SIMDE_TESTS_GENERATE_SYMBOL(SIMDE_TESTS_CURRENT_NEON_OP)(void) {
  static MunitSuite suite = { (char*) "/v" HEDLEY_STRINGIFY(SIMDE_TESTS_CURRENT_NEON_OP), vabd_tests, NULL, 1, MUNIT_SUITE_OPTION_NONE };

  return &suite;
}

HEDLEY_DIAGNOSTIC_POP
