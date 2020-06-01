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

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint32x2_t r = simde_vabd_u32(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_uint32x2(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}


static MunitResult
test_simde_vabd_f32(const MunitParameter params[], void* data) {
  (void) params;
  (void) data;

  const struct {
    simde_float32x2_t a;
    simde_float32x2_t b;
    simde_float32x2_t r;
  } test_vec[8] = {
    { simde_x_vload_f32(SIMDE_FLOAT32_C(  689.19), SIMDE_FLOAT32_C( -303.34)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  -66.67), SIMDE_FLOAT32_C( -401.68)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  755.86), SIMDE_FLOAT32_C(   98.34)) },
    { simde_x_vload_f32(SIMDE_FLOAT32_C( -939.36), SIMDE_FLOAT32_C(  983.73)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  949.74), SIMDE_FLOAT32_C( -718.64)),
      simde_x_vload_f32(SIMDE_FLOAT32_C( 1889.10), SIMDE_FLOAT32_C( 1702.37)) },
    { simde_x_vload_f32(SIMDE_FLOAT32_C( -431.59), SIMDE_FLOAT32_C(  -80.63)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  822.30), SIMDE_FLOAT32_C( -755.73)),
      simde_x_vload_f32(SIMDE_FLOAT32_C( 1253.89), SIMDE_FLOAT32_C(  675.10)) },
    { simde_x_vload_f32(SIMDE_FLOAT32_C(   -5.28), SIMDE_FLOAT32_C(  466.08)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  443.24), SIMDE_FLOAT32_C(  423.22)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  448.52), SIMDE_FLOAT32_C(   42.86)) },
    { simde_x_vload_f32(SIMDE_FLOAT32_C(  113.60), SIMDE_FLOAT32_C(  279.75)),
      simde_x_vload_f32(SIMDE_FLOAT32_C( -492.46), SIMDE_FLOAT32_C( -866.98)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  606.06), SIMDE_FLOAT32_C( 1146.73)) },
    { simde_x_vload_f32(SIMDE_FLOAT32_C(  928.97), SIMDE_FLOAT32_C(   97.55)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  966.19), SIMDE_FLOAT32_C(   43.80)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(   37.22), SIMDE_FLOAT32_C(   53.75)) },
    { simde_x_vload_f32(SIMDE_FLOAT32_C(  351.24), SIMDE_FLOAT32_C( -255.62)),
      simde_x_vload_f32(SIMDE_FLOAT32_C( -309.50), SIMDE_FLOAT32_C(  968.78)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  660.74), SIMDE_FLOAT32_C( 1224.40)) },
    { simde_x_vload_f32(SIMDE_FLOAT32_C( -699.58), SIMDE_FLOAT32_C(  814.65)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  171.28), SIMDE_FLOAT32_C(  -54.03)),
      simde_x_vload_f32(SIMDE_FLOAT32_C(  870.86), SIMDE_FLOAT32_C(  868.68)) },

  };

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
    { simde_x_vload_f64(SIMDE_FLOAT64_C(  689.19)),
      simde_x_vload_f64(SIMDE_FLOAT64_C( -303.34)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(  992.53)) },
    { simde_x_vload_f64(SIMDE_FLOAT64_C(  -66.67)),
      simde_x_vload_f64(SIMDE_FLOAT64_C( -401.68)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(  335.01)) },
    { simde_x_vload_f64(SIMDE_FLOAT64_C( -939.36)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(  983.73)),
      simde_x_vload_f64(SIMDE_FLOAT64_C( 1923.09)) },
    { simde_x_vload_f64(SIMDE_FLOAT64_C(  949.74)),
      simde_x_vload_f64(SIMDE_FLOAT64_C( -718.64)),
      simde_x_vload_f64(SIMDE_FLOAT64_C( 1668.38)) },
    { simde_x_vload_f64(SIMDE_FLOAT64_C( -431.59)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(  -80.63)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(  350.96)) },
    { simde_x_vload_f64(SIMDE_FLOAT64_C(  822.30)),
      simde_x_vload_f64(SIMDE_FLOAT64_C( -755.73)),
      simde_x_vload_f64(SIMDE_FLOAT64_C( 1578.03)) },
    { simde_x_vload_f64(SIMDE_FLOAT64_C(   -5.28)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(  466.08)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(  471.36)) },
    { simde_x_vload_f64(SIMDE_FLOAT64_C(  443.24)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(  423.22)),
      simde_x_vload_f64(SIMDE_FLOAT64_C(   20.02)) },

  };

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
    { simde_x_vloadq_s8(INT8_C(  14), INT8_C( -98), INT8_C( -16), INT8_C(  -2),
                        INT8_C( -17), INT8_C( -21), INT8_C( 117), INT8_C(-122),
                        INT8_C( 101), INT8_C(   3), INT8_C( 127), INT8_C(  42),
                        INT8_C(  66), INT8_C( -84), INT8_C( -72), INT8_C(  41)),
      simde_x_vloadq_s8(INT8_C( 101), INT8_C(  80), INT8_C(  22), INT8_C( -57),
                        INT8_C(  37), INT8_C( -53), INT8_C( -33), INT8_C( -66),
                        INT8_C( -20), INT8_C(-120), INT8_C(  84), INT8_C(   2),
                        INT8_C( -78), INT8_C( 105), INT8_C(-120), INT8_C(  14)),
      simde_x_vloadq_s8(INT8_C(  87), INT8_C( -78), INT8_C(  38), INT8_C(  55),
                        INT8_C(  54), INT8_C(  32), INT8_C(-106), INT8_C(  56),
                        INT8_C( 121), INT8_C( 123), INT8_C(  43), INT8_C(  40),
                        INT8_C(-112), INT8_C( -67), INT8_C(  48), INT8_C(  27)) },
    { simde_x_vloadq_s8(INT8_C(   5), INT8_C(  91), INT8_C( 109), INT8_C(  50),
                        INT8_C(  68), INT8_C(  67), INT8_C( 117), INT8_C(  75),
                        INT8_C(   8), INT8_C(  80), INT8_C(  45), INT8_C( -17),
                        INT8_C(  13), INT8_C( -41), INT8_C( -28), INT8_C(  42)),
      simde_x_vloadq_s8(INT8_C(-110), INT8_C( 102), INT8_C( -76), INT8_C( -50),
                        INT8_C( -12), INT8_C( 118), INT8_C( -24), INT8_C(  90),
                        INT8_C(   3), INT8_C( -82), INT8_C( -30), INT8_C( -60),
                        INT8_C( 101), INT8_C( -16), INT8_C(  93), INT8_C( -32)),
      simde_x_vloadq_s8(INT8_C( 115), INT8_C(  11), INT8_C( -71), INT8_C( 100),
                        INT8_C(  80), INT8_C(  51), INT8_C(-115), INT8_C(  15),
                        INT8_C(   5), INT8_C( -94), INT8_C(  75), INT8_C(  43),
                        INT8_C(  88), INT8_C(  25), INT8_C( 121), INT8_C(  74)) },
    { simde_x_vloadq_s8(INT8_C(  -9), INT8_C(  57), INT8_C( -38), INT8_C( -86),
                        INT8_C( -57), INT8_C(-101), INT8_C(  59), INT8_C(  94),
                        INT8_C(-111), INT8_C(  51), INT8_C(-117), INT8_C( -25),
                        INT8_C( -37), INT8_C( -92), INT8_C(  11), INT8_C( -12)),
      simde_x_vloadq_s8(INT8_C(  13), INT8_C( -12), INT8_C(  -3), INT8_C( -64),
                        INT8_C(-117), INT8_C( -11), INT8_C( -53), INT8_C(  66),
                        INT8_C( -71), INT8_C( -67), INT8_C( 122), INT8_C(  58),
                        INT8_C( 118), INT8_C(  58), INT8_C( -35), INT8_C(   2)),
      simde_x_vloadq_s8(INT8_C(  22), INT8_C(  69), INT8_C(  35), INT8_C(  22),
                        INT8_C(  60), INT8_C(  90), INT8_C( 112), INT8_C(  28),
                        INT8_C(  40), INT8_C( 118), INT8_C( -17), INT8_C(  83),
                        INT8_C(-101), INT8_C(-106), INT8_C(  46), INT8_C(  14)) },
    { simde_x_vloadq_s8(INT8_C( -76), INT8_C(  -9), INT8_C( -99), INT8_C(-101),
                        INT8_C(-125), INT8_C( -46), INT8_C(  85), INT8_C(  82),
                        INT8_C(  75), INT8_C( 119), INT8_C(-107), INT8_C(  77),
                        INT8_C( -48), INT8_C(  -8), INT8_C( -74), INT8_C(  71)),
      simde_x_vloadq_s8(INT8_C(  33), INT8_C(  45), INT8_C( -46), INT8_C(-119),
                        INT8_C(  54), INT8_C( -14), INT8_C(  60), INT8_C(  63),
                        INT8_C( 114), INT8_C( 119), INT8_C(  96), INT8_C( -94),
                        INT8_C(-114), INT8_C(  20), INT8_C( -97), INT8_C(  86)),
      simde_x_vloadq_s8(INT8_C( 109), INT8_C(  54), INT8_C(  53), INT8_C(  18),
                        INT8_C( -77), INT8_C(  32), INT8_C(  25), INT8_C(  19),
                        INT8_C(  39), INT8_C(   0), INT8_C( -53), INT8_C( -85),
                        INT8_C(  66), INT8_C(  28), INT8_C(  23), INT8_C(  15)) },
    { simde_x_vloadq_s8(INT8_C(  72), INT8_C(  54), INT8_C( -91), INT8_C( -60),
                        INT8_C( -13), INT8_C( -80), INT8_C( -48), INT8_C( -92),
                        INT8_C(  72), INT8_C(  43), INT8_C( -59), INT8_C( -87),
                        INT8_C( -61), INT8_C(-110), INT8_C( 116), INT8_C( -40)),
      simde_x_vloadq_s8(INT8_C(  38), INT8_C(  71), INT8_C(  -8), INT8_C(  51),
                        INT8_C(-122), INT8_C( -13), INT8_C(-113), INT8_C(  64),
                        INT8_C(  77), INT8_C(-102), INT8_C( -90), INT8_C(   6),
                        INT8_C( -62), INT8_C( -39), INT8_C( -23), INT8_C( -50)),
      simde_x_vloadq_s8(INT8_C(  34), INT8_C(  17), INT8_C(  83), INT8_C( 111),
                        INT8_C( 109), INT8_C(  67), INT8_C(  65), INT8_C(-100),
                        INT8_C(   5), INT8_C(-111), INT8_C(  31), INT8_C(  93),
                        INT8_C(   1), INT8_C(  71), INT8_C(-117), INT8_C(  10)) },
    { simde_x_vloadq_s8(INT8_C(  -9), INT8_C(  19), INT8_C(-122), INT8_C(  61),
                        INT8_C(  56), INT8_C( -88), INT8_C(  -7), INT8_C( 122),
                        INT8_C(   0), INT8_C(-112), INT8_C( -44), INT8_C(  57),
                        INT8_C(-117), INT8_C( -90), INT8_C(  13), INT8_C(  91)),
      simde_x_vloadq_s8(INT8_C(  38), INT8_C(-102), INT8_C( -34), INT8_C(-120),
                        INT8_C( -35), INT8_C( -72), INT8_C(   9), INT8_C(-111),
                        INT8_C(  90), INT8_C(  31), INT8_C( -26), INT8_C( -22),
                        INT8_C( -80), INT8_C( 112), INT8_C( -77), INT8_C(  -9)),
      simde_x_vloadq_s8(INT8_C(  47), INT8_C( 121), INT8_C(  88), INT8_C( -75),
                        INT8_C(  91), INT8_C(  16), INT8_C(  16), INT8_C( -23),
                        INT8_C(  90), INT8_C(-113), INT8_C(  18), INT8_C(  79),
                        INT8_C(  37), INT8_C( -54), INT8_C(  90), INT8_C( 100)) },
    { simde_x_vloadq_s8(INT8_C(  98), INT8_C(  67), INT8_C( -45), INT8_C(-128),
                        INT8_C(  10), INT8_C(  79), INT8_C(   5), INT8_C( -89),
                        INT8_C(-114), INT8_C( -62), INT8_C(  24), INT8_C( -92),
                        INT8_C(  56), INT8_C( -75), INT8_C( -66), INT8_C(  61)),
      simde_x_vloadq_s8(INT8_C(  23), INT8_C( -98), INT8_C(  -9), INT8_C(-109),
                        INT8_C(-126), INT8_C( -46), INT8_C(  -1), INT8_C(  39),
                        INT8_C( -48), INT8_C( 124), INT8_C( 122), INT8_C( -45),
                        INT8_C(-112), INT8_C( -19), INT8_C( 127), INT8_C( -88)),
      simde_x_vloadq_s8(INT8_C(  75), INT8_C( -91), INT8_C(  36), INT8_C(  19),
                        INT8_C(-120), INT8_C( 125), INT8_C(   6), INT8_C(-128),
                        INT8_C(  66), INT8_C( -70), INT8_C(  98), INT8_C(  47),
                        INT8_C( -88), INT8_C(  56), INT8_C( -63), INT8_C(-107)) },
    { simde_x_vloadq_s8(INT8_C( -86), INT8_C( -21), INT8_C( -99), INT8_C(-106),
                        INT8_C(-103), INT8_C(-123), INT8_C(  19), INT8_C(  63),
                        INT8_C( 116), INT8_C(  94), INT8_C(  79), INT8_C(  64),
                        INT8_C(  51), INT8_C( -43), INT8_C(  87), INT8_C(-123)),
      simde_x_vloadq_s8(INT8_C( -81), INT8_C(  18), INT8_C(  55), INT8_C(-118),
                        INT8_C(  59), INT8_C( -34), INT8_C( -80), INT8_C(  96),
                        INT8_C( -67), INT8_C( -66), INT8_C(-114), INT8_C( -53),
                        INT8_C( -70), INT8_C( -74), INT8_C(  79), INT8_C( -25)),
      simde_x_vloadq_s8(INT8_C(   5), INT8_C(  39), INT8_C(-102), INT8_C(  12),
                        INT8_C( -94), INT8_C(  89), INT8_C(  99), INT8_C(  33),
                        INT8_C( -73), INT8_C( -96), INT8_C( -63), INT8_C( 117),
                        INT8_C( 121), INT8_C(  31), INT8_C(   8), INT8_C(  98)) },

  };

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
    { simde_x_vloadq_s16(INT16_C(-25074), INT16_C(  -272), INT16_C( -5137), INT16_C(-31115),
                         INT16_C(   869), INT16_C( 10879), INT16_C(-21438), INT16_C( 10680)),
      simde_x_vloadq_s16(INT16_C( 20581), INT16_C(-14570), INT16_C(-13531), INT16_C(-16673),
                         INT16_C(-30484), INT16_C(   596), INT16_C( 27058), INT16_C(  3720)),
      simde_x_vloadq_s16(INT16_C(-19881), INT16_C( 14298), INT16_C(  8394), INT16_C( 14442),
                         INT16_C( 31353), INT16_C( 10283), INT16_C(-17040), INT16_C(  6960)) },
    { simde_x_vloadq_s16(INT16_C( 23301), INT16_C( 12909), INT16_C( 17220), INT16_C( 19317),
                         INT16_C( 20488), INT16_C( -4307), INT16_C(-10483), INT16_C( 10980)),
      simde_x_vloadq_s16(INT16_C( 26258), INT16_C(-12620), INT16_C( 30452), INT16_C( 23272),
                         INT16_C(-20989), INT16_C(-15134), INT16_C( -3995), INT16_C( -8099)),
      simde_x_vloadq_s16(INT16_C(  2957), INT16_C( 25529), INT16_C( 13232), INT16_C(  3955),
                         INT16_C(-24059), INT16_C( 10827), INT16_C(  6488), INT16_C( 19079)) },
    { simde_x_vloadq_s16(INT16_C( 14839), INT16_C(-21798), INT16_C(-25657), INT16_C( 24123),
                         INT16_C( 13201), INT16_C( -6261), INT16_C(-23333), INT16_C( -3061)),
      simde_x_vloadq_s16(INT16_C( -3059), INT16_C(-16131), INT16_C( -2677), INT16_C( 17099),
                         INT16_C(-16967), INT16_C( 14970), INT16_C( 14966), INT16_C(   733)),
      simde_x_vloadq_s16(INT16_C( 17898), INT16_C(  5667), INT16_C( 22980), INT16_C(  7024),
                         INT16_C( 30168), INT16_C( 21231), INT16_C(-27237), INT16_C(  3794)) },
    { simde_x_vloadq_s16(INT16_C( -2124), INT16_C(-25699), INT16_C(-11645), INT16_C( 21077),
                         INT16_C( 30539), INT16_C( 19861), INT16_C( -1840), INT16_C( 18358)),
      simde_x_vloadq_s16(INT16_C( 11553), INT16_C(-30254), INT16_C( -3530), INT16_C( 16188),
                         INT16_C( 30578), INT16_C(-23968), INT16_C(  5262), INT16_C( 22175)),
      simde_x_vloadq_s16(INT16_C( 13677), INT16_C(  4555), INT16_C(  8115), INT16_C(  4889),
                         INT16_C(    39), INT16_C(-21707), INT16_C(  7102), INT16_C(  3817)) },
    { simde_x_vloadq_s16(INT16_C( 13896), INT16_C(-15195), INT16_C(-20237), INT16_C(-23344),
                         INT16_C( 11080), INT16_C(-22075), INT16_C(-27965), INT16_C(-10124)),
      simde_x_vloadq_s16(INT16_C( 18214), INT16_C( 13304), INT16_C( -3194), INT16_C( 16527),
                         INT16_C(-26035), INT16_C(  1702), INT16_C( -9790), INT16_C(-12567)),
      simde_x_vloadq_s16(INT16_C(  4318), INT16_C( 28499), INT16_C( 17043), INT16_C(-25665),
                         INT16_C(-28421), INT16_C( 23777), INT16_C( 18175), INT16_C(  2443)) },
    { simde_x_vloadq_s16(INT16_C(  5111), INT16_C( 15750), INT16_C(-22472), INT16_C( 31481),
                         INT16_C(-28672), INT16_C( 14804), INT16_C(-22901), INT16_C( 23309)),
      simde_x_vloadq_s16(INT16_C(-26074), INT16_C(-30498), INT16_C(-18211), INT16_C(-28407),
                         INT16_C(  8026), INT16_C( -5402), INT16_C( 28848), INT16_C( -2125)),
      simde_x_vloadq_s16(INT16_C( 31185), INT16_C(-19288), INT16_C(  4261), INT16_C( -5648),
                         INT16_C(-28838), INT16_C( 20206), INT16_C(-13787), INT16_C( 25434)) },
    { simde_x_vloadq_s16(INT16_C( 17250), INT16_C(-32557), INT16_C( 20234), INT16_C(-22779),
                         INT16_C(-15730), INT16_C(-23528), INT16_C(-19144), INT16_C( 15806)),
      simde_x_vloadq_s16(INT16_C(-25065), INT16_C(-27657), INT16_C(-11646), INT16_C( 10239),
                         INT16_C( 31952), INT16_C(-11398), INT16_C( -4720), INT16_C(-22401)),
      simde_x_vloadq_s16(INT16_C(-23221), INT16_C(  4900), INT16_C( 31880), INT16_C(-32518),
                         INT16_C(-17854), INT16_C( 12130), INT16_C( 14424), INT16_C(-27329)) },
    { simde_x_vloadq_s16(INT16_C( -5206), INT16_C(-26979), INT16_C(-31335), INT16_C( 16147),
                         INT16_C( 24180), INT16_C( 16463), INT16_C(-10957), INT16_C(-31401)),
      simde_x_vloadq_s16(INT16_C(  4783), INT16_C(-30153), INT16_C( -8645), INT16_C( 24752),
                         INT16_C(-16707), INT16_C(-13426), INT16_C(-18758), INT16_C( -6321)),
      simde_x_vloadq_s16(INT16_C(  9989), INT16_C(  3174), INT16_C( 22690), INT16_C(  8605),
                         INT16_C(-24649), INT16_C( 29889), INT16_C(  7801), INT16_C( 25080)) },

  };

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
    { simde_x_vloadq_s32(INT32_C(  -17785330), INT32_C(-2039092241),
                         INT32_C(  712967013), INT32_C(  699968578)),
      simde_x_vloadq_s32(INT32_C( -954838939), INT32_C(-1092629723),
                         INT32_C(   39094508), INT32_C(  243820978)),
      simde_x_vloadq_s32(INT32_C(  937053609), INT32_C(  946462518),
                         INT32_C(  673872505), INT32_C(  456147600)) },
    { simde_x_vloadq_s32(INT32_C(  846027525), INT32_C( 1265976132),
                         INT32_C( -282243064), INT32_C(  719640333)),
      simde_x_vloadq_s32(INT32_C( -827038062), INT32_C( 1525184244),
                         INT32_C( -991777277), INT32_C( -530714523)),
      simde_x_vloadq_s32(INT32_C( 1673065587), INT32_C(  259208112),
                         INT32_C(  709534213), INT32_C( 1250354856)) },
    { simde_x_vloadq_s32(INT32_C(-1428538889), INT32_C( 1580964807),
                         INT32_C( -410307695), INT32_C( -200563493)),
      simde_x_vloadq_s32(INT32_C(-1057098739), INT32_C( 1120662923),
                         INT32_C(  981122489), INT32_C(   48052854)),
      simde_x_vloadq_s32(INT32_C(  371440150), INT32_C(  460301884),
                         INT32_C( 1391430184), INT32_C(  248616347)) },
    { simde_x_vloadq_s32(INT32_C(-1684146252), INT32_C( 1381356163),
                         INT32_C( 1301641035), INT32_C( 1203173584)),
      simde_x_vloadq_s32(INT32_C(-1982714591), INT32_C( 1060958774),
                         INT32_C(-1570736270), INT32_C( 1453266062)),
      simde_x_vloadq_s32(INT32_C(  298568339), INT32_C(  320397389),
                         INT32_C(-1422589991), INT32_C(  250092478)) },
    { simde_x_vloadq_s32(INT32_C( -995805624), INT32_C(-1529827085),
                         INT32_C(-1446696120), INT32_C( -663448893)),
      simde_x_vloadq_s32(INT32_C(  871909158), INT32_C( 1083175814),
                         INT32_C(  111581773), INT32_C( -823535166)),
      simde_x_vloadq_s32(INT32_C( 1867714782), INT32_C(-1681964397),
                         INT32_C( 1558277893), INT32_C(  160086273)) },
    { simde_x_vloadq_s32(INT32_C( 1032197111), INT32_C( 2063181880),
                         INT32_C(  970231808), INT32_C( 1527621259)),
      simde_x_vloadq_s32(INT32_C(-1998677466), INT32_C(-1861633827),
                         INT32_C( -354017446), INT32_C( -139235152)),
      simde_x_vloadq_s32(INT32_C(-1264092719), INT32_C( -370151589),
                         INT32_C( 1324249254), INT32_C( 1666856411)) },
    { simde_x_vloadq_s32(INT32_C(-2133638302), INT32_C(-1492824310),
                         INT32_C(-1541881202), INT32_C( 1035908408)),
      simde_x_vloadq_s32(INT32_C(-1812488681), INT32_C(  671076994),
                         INT32_C( -746947376), INT32_C(-1468011120)),
      simde_x_vloadq_s32(INT32_C(  321149621), INT32_C(-2131065992),
                         INT32_C(  794933826), INT32_C(-1791047768)) },
    { simde_x_vloadq_s32(INT32_C(-1768035414), INT32_C( 1058243993),
                         INT32_C( 1078943348), INT32_C(-2057841357)),
      simde_x_vloadq_s32(INT32_C(-1976102225), INT32_C( 1622203963),
                         INT32_C( -879837507), INT32_C( -414206278)),
      simde_x_vloadq_s32(INT32_C(  208066811), INT32_C(  563959970),
                         INT32_C( 1958780855), INT32_C( 1643635079)) },

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_int32x4_t r = simde_vabdq_s32(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_int32x4(r, ==, test_vec[i].r);
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
    { simde_x_vloadq_u8(UINT8_C( 14), UINT8_C(158), UINT8_C(240), UINT8_C(254),
                        UINT8_C(239), UINT8_C(235), UINT8_C(117), UINT8_C(134),
                        UINT8_C(101), UINT8_C(  3), UINT8_C(127), UINT8_C( 42),
                        UINT8_C( 66), UINT8_C(172), UINT8_C(184), UINT8_C( 41) ),
      simde_x_vloadq_u8(UINT8_C(101), UINT8_C( 80), UINT8_C( 22), UINT8_C(199),
                        UINT8_C( 37), UINT8_C(203), UINT8_C(223), UINT8_C(190),
                        UINT8_C(236), UINT8_C(136), UINT8_C( 84), UINT8_C(  2),
                        UINT8_C(178), UINT8_C(105), UINT8_C(136), UINT8_C( 14) ),
      simde_x_vloadq_u8(UINT8_C( 87), UINT8_C( 78), UINT8_C(218), UINT8_C( 55),
                        UINT8_C(202), UINT8_C( 32), UINT8_C(106), UINT8_C( 56),
                        UINT8_C(135), UINT8_C(133), UINT8_C( 43), UINT8_C( 40),
                        UINT8_C(112), UINT8_C( 67), UINT8_C( 48), UINT8_C( 27) ) },
    { simde_x_vloadq_u8(UINT8_C(  5), UINT8_C( 91), UINT8_C(109), UINT8_C( 50),
                        UINT8_C( 68), UINT8_C( 67), UINT8_C(117), UINT8_C( 75),
                        UINT8_C(  8), UINT8_C( 80), UINT8_C( 45), UINT8_C(239),
                        UINT8_C( 13), UINT8_C(215), UINT8_C(228), UINT8_C( 42) ),
      simde_x_vloadq_u8(UINT8_C(146), UINT8_C(102), UINT8_C(180), UINT8_C(206),
                        UINT8_C(244), UINT8_C(118), UINT8_C(232), UINT8_C( 90),
                        UINT8_C(  3), UINT8_C(174), UINT8_C(226), UINT8_C(196),
                        UINT8_C(101), UINT8_C(240), UINT8_C( 93), UINT8_C(224) ),
      simde_x_vloadq_u8(UINT8_C(141), UINT8_C( 11), UINT8_C( 71), UINT8_C(156),
                        UINT8_C(176), UINT8_C( 51), UINT8_C(115), UINT8_C( 15),
                        UINT8_C(  5), UINT8_C( 94), UINT8_C(181), UINT8_C( 43),
                        UINT8_C( 88), UINT8_C( 25), UINT8_C(135), UINT8_C(182) ) },
    { simde_x_vloadq_u8(UINT8_C(247), UINT8_C( 57), UINT8_C(218), UINT8_C(170),
                        UINT8_C(199), UINT8_C(155), UINT8_C( 59), UINT8_C( 94),
                        UINT8_C(145), UINT8_C( 51), UINT8_C(139), UINT8_C(231),
                        UINT8_C(219), UINT8_C(164), UINT8_C( 11), UINT8_C(244) ),
      simde_x_vloadq_u8(UINT8_C( 13), UINT8_C(244), UINT8_C(253), UINT8_C(192),
                        UINT8_C(139), UINT8_C(245), UINT8_C(203), UINT8_C( 66),
                        UINT8_C(185), UINT8_C(189), UINT8_C(122), UINT8_C( 58),
                        UINT8_C(118), UINT8_C( 58), UINT8_C(221), UINT8_C(  2) ),
      simde_x_vloadq_u8(UINT8_C(234), UINT8_C(187), UINT8_C( 35), UINT8_C( 22),
                        UINT8_C( 60), UINT8_C( 90), UINT8_C(144), UINT8_C( 28),
                        UINT8_C( 40), UINT8_C(138), UINT8_C( 17), UINT8_C(173),
                        UINT8_C(101), UINT8_C(106), UINT8_C(210), UINT8_C(242) ) },
    { simde_x_vloadq_u8(UINT8_C(180), UINT8_C(247), UINT8_C(157), UINT8_C(155),
                        UINT8_C(131), UINT8_C(210), UINT8_C( 85), UINT8_C( 82),
                        UINT8_C( 75), UINT8_C(119), UINT8_C(149), UINT8_C( 77),
                        UINT8_C(208), UINT8_C(248), UINT8_C(182), UINT8_C( 71) ),
      simde_x_vloadq_u8(UINT8_C( 33), UINT8_C( 45), UINT8_C(210), UINT8_C(137),
                        UINT8_C( 54), UINT8_C(242), UINT8_C( 60), UINT8_C( 63),
                        UINT8_C(114), UINT8_C(119), UINT8_C( 96), UINT8_C(162),
                        UINT8_C(142), UINT8_C( 20), UINT8_C(159), UINT8_C( 86) ),
      simde_x_vloadq_u8(UINT8_C(147), UINT8_C(202), UINT8_C( 53), UINT8_C( 18),
                        UINT8_C( 77), UINT8_C( 32), UINT8_C( 25), UINT8_C( 19),
                        UINT8_C( 39), UINT8_C(  0), UINT8_C( 53), UINT8_C( 85),
                        UINT8_C( 66), UINT8_C(228), UINT8_C( 23), UINT8_C( 15) ) },
    { simde_x_vloadq_u8(UINT8_C( 72), UINT8_C( 54), UINT8_C(165), UINT8_C(196),
                        UINT8_C(243), UINT8_C(176), UINT8_C(208), UINT8_C(164),
                        UINT8_C( 72), UINT8_C( 43), UINT8_C(197), UINT8_C(169),
                        UINT8_C(195), UINT8_C(146), UINT8_C(116), UINT8_C(216) ),
      simde_x_vloadq_u8(UINT8_C( 38), UINT8_C( 71), UINT8_C(248), UINT8_C( 51),
                        UINT8_C(134), UINT8_C(243), UINT8_C(143), UINT8_C( 64),
                        UINT8_C( 77), UINT8_C(154), UINT8_C(166), UINT8_C(  6),
                        UINT8_C(194), UINT8_C(217), UINT8_C(233), UINT8_C(206) ),
      simde_x_vloadq_u8(UINT8_C( 34), UINT8_C( 17), UINT8_C( 83), UINT8_C(145),
                        UINT8_C(109), UINT8_C( 67), UINT8_C( 65), UINT8_C(100),
                        UINT8_C(  5), UINT8_C(111), UINT8_C( 31), UINT8_C(163),
                        UINT8_C(  1), UINT8_C( 71), UINT8_C(117), UINT8_C( 10) ) },
    { simde_x_vloadq_u8(UINT8_C(247), UINT8_C( 19), UINT8_C(134), UINT8_C( 61),
                        UINT8_C( 56), UINT8_C(168), UINT8_C(249), UINT8_C(122),
                        UINT8_C(  0), UINT8_C(144), UINT8_C(212), UINT8_C( 57),
                        UINT8_C(139), UINT8_C(166), UINT8_C( 13), UINT8_C( 91) ),
      simde_x_vloadq_u8(UINT8_C( 38), UINT8_C(154), UINT8_C(222), UINT8_C(136),
                        UINT8_C(221), UINT8_C(184), UINT8_C(  9), UINT8_C(145),
                        UINT8_C( 90), UINT8_C( 31), UINT8_C(230), UINT8_C(234),
                        UINT8_C(176), UINT8_C(112), UINT8_C(179), UINT8_C(247) ),
      simde_x_vloadq_u8(UINT8_C(209), UINT8_C(135), UINT8_C( 88), UINT8_C( 75),
                        UINT8_C(165), UINT8_C( 16), UINT8_C(240), UINT8_C( 23),
                        UINT8_C( 90), UINT8_C(113), UINT8_C( 18), UINT8_C(177),
                        UINT8_C( 37), UINT8_C( 54), UINT8_C(166), UINT8_C(156) ) },
    { simde_x_vloadq_u8(UINT8_C( 98), UINT8_C( 67), UINT8_C(211), UINT8_C(128),
                        UINT8_C( 10), UINT8_C( 79), UINT8_C(  5), UINT8_C(167),
                        UINT8_C(142), UINT8_C(194), UINT8_C( 24), UINT8_C(164),
                        UINT8_C( 56), UINT8_C(181), UINT8_C(190), UINT8_C( 61) ),
      simde_x_vloadq_u8(UINT8_C( 23), UINT8_C(158), UINT8_C(247), UINT8_C(147),
                        UINT8_C(130), UINT8_C(210), UINT8_C(255), UINT8_C( 39),
                        UINT8_C(208), UINT8_C(124), UINT8_C(122), UINT8_C(211),
                        UINT8_C(144), UINT8_C(237), UINT8_C(127), UINT8_C(168) ),
      simde_x_vloadq_u8(UINT8_C( 75), UINT8_C( 91), UINT8_C( 36), UINT8_C( 19),
                        UINT8_C(120), UINT8_C(131), UINT8_C(250), UINT8_C(128),
                        UINT8_C( 66), UINT8_C( 70), UINT8_C( 98), UINT8_C( 47),
                        UINT8_C( 88), UINT8_C( 56), UINT8_C( 63), UINT8_C(107) ) },
    { simde_x_vloadq_u8(UINT8_C(170), UINT8_C(235), UINT8_C(157), UINT8_C(150),
                        UINT8_C(153), UINT8_C(133), UINT8_C( 19), UINT8_C( 63),
                        UINT8_C(116), UINT8_C( 94), UINT8_C( 79), UINT8_C( 64),
                        UINT8_C( 51), UINT8_C(213), UINT8_C( 87), UINT8_C(133) ),
      simde_x_vloadq_u8(UINT8_C(175), UINT8_C( 18), UINT8_C( 55), UINT8_C(138),
                        UINT8_C( 59), UINT8_C(222), UINT8_C(176), UINT8_C( 96),
                        UINT8_C(189), UINT8_C(190), UINT8_C(142), UINT8_C(203),
                        UINT8_C(186), UINT8_C(182), UINT8_C( 79), UINT8_C(231) ),
      simde_x_vloadq_u8(UINT8_C(  5), UINT8_C(217), UINT8_C(102), UINT8_C( 12),
                        UINT8_C( 94), UINT8_C( 89), UINT8_C(157), UINT8_C( 33),
                        UINT8_C( 73), UINT8_C( 96), UINT8_C( 63), UINT8_C(139),
                        UINT8_C(135), UINT8_C( 31), UINT8_C(  8), UINT8_C( 98) ) },

  };

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
    { simde_x_vloadq_u16(UINT16_C(40462), UINT16_C(65264), UINT16_C(60399), UINT16_C(34421),
                         UINT16_C(  869), UINT16_C(10879), UINT16_C(44098), UINT16_C(10680)),
      simde_x_vloadq_u16(UINT16_C(20581), UINT16_C(50966), UINT16_C(52005), UINT16_C(48863),
                         UINT16_C(35052), UINT16_C(  596), UINT16_C(27058), UINT16_C( 3720)),
      simde_x_vloadq_u16(UINT16_C(19881), UINT16_C(14298), UINT16_C( 8394), UINT16_C(14442),
                         UINT16_C(34183), UINT16_C(10283), UINT16_C(17040), UINT16_C( 6960)) },
    { simde_x_vloadq_u16(UINT16_C(23301), UINT16_C(12909), UINT16_C(17220), UINT16_C(19317),
                         UINT16_C(20488), UINT16_C(61229), UINT16_C(55053), UINT16_C(10980)),
      simde_x_vloadq_u16(UINT16_C(26258), UINT16_C(52916), UINT16_C(30452), UINT16_C(23272),
                         UINT16_C(44547), UINT16_C(50402), UINT16_C(61541), UINT16_C(57437)),
      simde_x_vloadq_u16(UINT16_C( 2957), UINT16_C(40007), UINT16_C(13232), UINT16_C( 3955),
                         UINT16_C(24059), UINT16_C(10827), UINT16_C( 6488), UINT16_C(46457)) },
    { simde_x_vloadq_u16(UINT16_C(14839), UINT16_C(43738), UINT16_C(39879), UINT16_C(24123),
                         UINT16_C(13201), UINT16_C(59275), UINT16_C(42203), UINT16_C(62475)),
      simde_x_vloadq_u16(UINT16_C(62477), UINT16_C(49405), UINT16_C(62859), UINT16_C(17099),
                         UINT16_C(48569), UINT16_C(14970), UINT16_C(14966), UINT16_C(  733)),
      simde_x_vloadq_u16(UINT16_C(47638), UINT16_C( 5667), UINT16_C(22980), UINT16_C( 7024),
                         UINT16_C(35368), UINT16_C(44305), UINT16_C(27237), UINT16_C(61742)) },
    { simde_x_vloadq_u16(UINT16_C(63412), UINT16_C(39837), UINT16_C(53891), UINT16_C(21077),
                         UINT16_C(30539), UINT16_C(19861), UINT16_C(63696), UINT16_C(18358)),
      simde_x_vloadq_u16(UINT16_C(11553), UINT16_C(35282), UINT16_C(62006), UINT16_C(16188),
                         UINT16_C(30578), UINT16_C(41568), UINT16_C( 5262), UINT16_C(22175)),
      simde_x_vloadq_u16(UINT16_C(51859), UINT16_C( 4555), UINT16_C( 8115), UINT16_C( 4889),
                         UINT16_C(   39), UINT16_C(21707), UINT16_C(58434), UINT16_C( 3817)) },
    { simde_x_vloadq_u16(UINT16_C(13896), UINT16_C(50341), UINT16_C(45299), UINT16_C(42192),
                         UINT16_C(11080), UINT16_C(43461), UINT16_C(37571), UINT16_C(55412)),
      simde_x_vloadq_u16(UINT16_C(18214), UINT16_C(13304), UINT16_C(62342), UINT16_C(16527),
                         UINT16_C(39501), UINT16_C( 1702), UINT16_C(55746), UINT16_C(52969)),
      simde_x_vloadq_u16(UINT16_C( 4318), UINT16_C(37037), UINT16_C(17043), UINT16_C(25665),
                         UINT16_C(28421), UINT16_C(41759), UINT16_C(18175), UINT16_C( 2443)) },
    { simde_x_vloadq_u16(UINT16_C( 5111), UINT16_C(15750), UINT16_C(43064), UINT16_C(31481),
                         UINT16_C(36864), UINT16_C(14804), UINT16_C(42635), UINT16_C(23309)),
      simde_x_vloadq_u16(UINT16_C(39462), UINT16_C(35038), UINT16_C(47325), UINT16_C(37129),
                         UINT16_C( 8026), UINT16_C(60134), UINT16_C(28848), UINT16_C(63411)),
      simde_x_vloadq_u16(UINT16_C(34351), UINT16_C(19288), UINT16_C( 4261), UINT16_C( 5648),
                         UINT16_C(28838), UINT16_C(45330), UINT16_C(13787), UINT16_C(40102)) },
    { simde_x_vloadq_u16(UINT16_C(17250), UINT16_C(32979), UINT16_C(20234), UINT16_C(42757),
                         UINT16_C(49806), UINT16_C(42008), UINT16_C(46392), UINT16_C(15806)),
      simde_x_vloadq_u16(UINT16_C(40471), UINT16_C(37879), UINT16_C(53890), UINT16_C(10239),
                         UINT16_C(31952), UINT16_C(54138), UINT16_C(60816), UINT16_C(43135)),
      simde_x_vloadq_u16(UINT16_C(23221), UINT16_C( 4900), UINT16_C(33656), UINT16_C(32518),
                         UINT16_C(17854), UINT16_C(12130), UINT16_C(14424), UINT16_C(27329)) },
    { simde_x_vloadq_u16(UINT16_C(60330), UINT16_C(38557), UINT16_C(34201), UINT16_C(16147),
                         UINT16_C(24180), UINT16_C(16463), UINT16_C(54579), UINT16_C(34135)),
      simde_x_vloadq_u16(UINT16_C( 4783), UINT16_C(35383), UINT16_C(56891), UINT16_C(24752),
                         UINT16_C(48829), UINT16_C(52110), UINT16_C(46778), UINT16_C(59215)),
      simde_x_vloadq_u16(UINT16_C(55547), UINT16_C( 3174), UINT16_C(22690), UINT16_C( 8605),
                         UINT16_C(24649), UINT16_C(35647), UINT16_C( 7801), UINT16_C(25080)) },

  };

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
    { simde_x_vloadq_u32(UINT32_C(4277181966), UINT32_C(2255875055),
                         UINT32_C( 712967013), UINT32_C( 699968578)),
      simde_x_vloadq_u32(UINT32_C(3340128357), UINT32_C(3202337573),
                         UINT32_C(  39094508), UINT32_C( 243820978)),
      simde_x_vloadq_u32(UINT32_C( 937053609), UINT32_C( 946462518),
                         UINT32_C( 673872505), UINT32_C( 456147600)) },
    { simde_x_vloadq_u32(UINT32_C( 846027525), UINT32_C(1265976132),
                         UINT32_C(4012724232), UINT32_C( 719640333)),
      simde_x_vloadq_u32(UINT32_C(3467929234), UINT32_C(1525184244),
                         UINT32_C(3303190019), UINT32_C(3764252773)),
      simde_x_vloadq_u32(UINT32_C(2621901709), UINT32_C( 259208112),
                         UINT32_C( 709534213), UINT32_C(3044612440)) },
    { simde_x_vloadq_u32(UINT32_C(2866428407), UINT32_C(1580964807),
                         UINT32_C(3884659601), UINT32_C(4094403803)),
      simde_x_vloadq_u32(UINT32_C(3237868557), UINT32_C(1120662923),
                         UINT32_C( 981122489), UINT32_C(  48052854)),
      simde_x_vloadq_u32(UINT32_C( 371440150), UINT32_C( 460301884),
                         UINT32_C(2903537112), UINT32_C(4046350949)) },
    { simde_x_vloadq_u32(UINT32_C(2610821044), UINT32_C(1381356163),
                         UINT32_C(1301641035), UINT32_C(1203173584)),
      simde_x_vloadq_u32(UINT32_C(2312252705), UINT32_C(1060958774),
                         UINT32_C(2724231026), UINT32_C(1453266062)),
      simde_x_vloadq_u32(UINT32_C( 298568339), UINT32_C( 320397389),
                         UINT32_C(1422589991), UINT32_C( 250092478)) },
    { simde_x_vloadq_u32(UINT32_C(3299161672), UINT32_C(2765140211),
                         UINT32_C(2848271176), UINT32_C(3631518403)),
      simde_x_vloadq_u32(UINT32_C( 871909158), UINT32_C(1083175814),
                         UINT32_C( 111581773), UINT32_C(3471432130)),
      simde_x_vloadq_u32(UINT32_C(2427252514), UINT32_C(1681964397),
                         UINT32_C(2736689403), UINT32_C( 160086273)) },
    { simde_x_vloadq_u32(UINT32_C(1032197111), UINT32_C(2063181880),
                         UINT32_C( 970231808), UINT32_C(1527621259)),
      simde_x_vloadq_u32(UINT32_C(2296289830), UINT32_C(2433333469),
                         UINT32_C(3940949850), UINT32_C(4155732144)),
      simde_x_vloadq_u32(UINT32_C(1264092719), UINT32_C( 370151589),
                         UINT32_C(2970718042), UINT32_C(2628110885)) },
    { simde_x_vloadq_u32(UINT32_C(2161328994), UINT32_C(2802142986),
                         UINT32_C(2753086094), UINT32_C(1035908408)),
      simde_x_vloadq_u32(UINT32_C(2482478615), UINT32_C( 671076994),
                         UINT32_C(3548019920), UINT32_C(2826956176)),
      simde_x_vloadq_u32(UINT32_C( 321149621), UINT32_C(2131065992),
                         UINT32_C( 794933826), UINT32_C(1791047768)) },
    { simde_x_vloadq_u32(UINT32_C(2526931882), UINT32_C(1058243993),
                         UINT32_C(1078943348), UINT32_C(2237125939)),
      simde_x_vloadq_u32(UINT32_C(2318865071), UINT32_C(1622203963),
                         UINT32_C(3415129789), UINT32_C(3880761018)),
      simde_x_vloadq_u32(UINT32_C( 208066811), UINT32_C( 563959970),
                         UINT32_C(2336186441), UINT32_C(1643635079)) },

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint32x4_t r = simde_vabdq_u32(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_uint32x4(r, ==, test_vec[i].r);
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
    { simde_x_vloadq_f32(SIMDE_FLOAT32_C(  809.06), SIMDE_FLOAT32_C(  792.36),
                         SIMDE_FLOAT32_C( -151.88), SIMDE_FLOAT32_C(  133.98)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  642.01), SIMDE_FLOAT32_C( -273.66),
                         SIMDE_FLOAT32_C( -754.30), SIMDE_FLOAT32_C( -813.46)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  167.05), SIMDE_FLOAT32_C( 1066.02),
                         SIMDE_FLOAT32_C(  602.42), SIMDE_FLOAT32_C(  947.44)) },
    { simde_x_vloadq_f32(SIMDE_FLOAT32_C( -220.83), SIMDE_FLOAT32_C( -469.27),
                         SIMDE_FLOAT32_C(  757.44), SIMDE_FLOAT32_C(  154.41)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C( -553.06), SIMDE_FLOAT32_C(  829.15),
                         SIMDE_FLOAT32_C( -679.80), SIMDE_FLOAT32_C( -243.74)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  332.23), SIMDE_FLOAT32_C( 1298.42),
                         SIMDE_FLOAT32_C( 1437.24), SIMDE_FLOAT32_C(  398.15)) },
    { simde_x_vloadq_f32(SIMDE_FLOAT32_C(  999.88), SIMDE_FLOAT32_C(  445.78),
                         SIMDE_FLOAT32_C(  341.40), SIMDE_FLOAT32_C( -293.17)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  471.36), SIMDE_FLOAT32_C( -272.46),
                         SIMDE_FLOAT32_C(  910.81), SIMDE_FLOAT32_C(  911.89)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  528.52), SIMDE_FLOAT32_C(  718.24),
                         SIMDE_FLOAT32_C(  569.41), SIMDE_FLOAT32_C( 1205.06)) },
    { simde_x_vloadq_f32(SIMDE_FLOAT32_C( -406.43), SIMDE_FLOAT32_C(  -70.51),
                         SIMDE_FLOAT32_C( -775.79), SIMDE_FLOAT32_C(  949.15)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(   83.98), SIMDE_FLOAT32_C(  583.58),
                         SIMDE_FLOAT32_C(  -32.22), SIMDE_FLOAT32_C(  -55.64)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  490.41), SIMDE_FLOAT32_C(  654.09),
                         SIMDE_FLOAT32_C(  743.57), SIMDE_FLOAT32_C( 1004.79)) },
    { simde_x_vloadq_f32(SIMDE_FLOAT32_C( -256.50), SIMDE_FLOAT32_C( -160.25),
                         SIMDE_FLOAT32_C(  173.41), SIMDE_FLOAT32_C(  754.36)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C( -426.99), SIMDE_FLOAT32_C(  645.30),
                         SIMDE_FLOAT32_C(  712.81), SIMDE_FLOAT32_C( -697.36)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  170.49), SIMDE_FLOAT32_C(  805.55),
                         SIMDE_FLOAT32_C(  539.40), SIMDE_FLOAT32_C( 1451.72)) },
    { simde_x_vloadq_f32(SIMDE_FLOAT32_C(  811.35), SIMDE_FLOAT32_C( -771.16),
                         SIMDE_FLOAT32_C(  347.51), SIMDE_FLOAT32_C(  -97.52)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C( -693.44), SIMDE_FLOAT32_C( -760.46),
                         SIMDE_FLOAT32_C(  769.64), SIMDE_FLOAT32_C(  240.65)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C( 1504.79), SIMDE_FLOAT32_C(   10.70),
                         SIMDE_FLOAT32_C(  422.13), SIMDE_FLOAT32_C(  338.17)) },
    { simde_x_vloadq_f32(SIMDE_FLOAT32_C(  163.33), SIMDE_FLOAT32_C(   43.82),
                         SIMDE_FLOAT32_C(   23.66), SIMDE_FLOAT32_C( -372.81)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  860.25), SIMDE_FLOAT32_C(  168.84),
                         SIMDE_FLOAT32_C(  714.53), SIMDE_FLOAT32_C( -652.76)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  696.92), SIMDE_FLOAT32_C(  125.02),
                         SIMDE_FLOAT32_C(  690.87), SIMDE_FLOAT32_C(  279.95)) },
    { simde_x_vloadq_f32(SIMDE_FLOAT32_C( -211.43), SIMDE_FLOAT32_C( -586.28),
                         SIMDE_FLOAT32_C(  864.71), SIMDE_FLOAT32_C( -665.46)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  170.31), SIMDE_FLOAT32_C( -349.99),
                         SIMDE_FLOAT32_C( -159.20), SIMDE_FLOAT32_C( -916.00)),
      simde_x_vloadq_f32(SIMDE_FLOAT32_C(  381.74), SIMDE_FLOAT32_C(  236.29),
                         SIMDE_FLOAT32_C( 1023.91), SIMDE_FLOAT32_C(  250.54)) },

  };

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
    { simde_x_vloadq_f64(SIMDE_FLOAT64_C(  809.06), SIMDE_FLOAT64_C(  792.36)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C( -151.88), SIMDE_FLOAT64_C(  133.98)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  960.94), SIMDE_FLOAT64_C(  658.38)) },
    { simde_x_vloadq_f64(SIMDE_FLOAT64_C(  642.01), SIMDE_FLOAT64_C( -273.66)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C( -754.30), SIMDE_FLOAT64_C( -813.46)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C( 1396.31), SIMDE_FLOAT64_C(  539.80)) },
    { simde_x_vloadq_f64(SIMDE_FLOAT64_C( -220.83), SIMDE_FLOAT64_C( -469.27)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  757.44), SIMDE_FLOAT64_C(  154.41)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  978.27), SIMDE_FLOAT64_C(  623.68)) },
    { simde_x_vloadq_f64(SIMDE_FLOAT64_C( -553.06), SIMDE_FLOAT64_C(  829.15)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C( -679.80), SIMDE_FLOAT64_C( -243.74)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  126.74), SIMDE_FLOAT64_C( 1072.89)) },
    { simde_x_vloadq_f64(SIMDE_FLOAT64_C(  999.88), SIMDE_FLOAT64_C(  445.78)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  341.40), SIMDE_FLOAT64_C( -293.17)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  658.48), SIMDE_FLOAT64_C(  738.95)) },
    { simde_x_vloadq_f64(SIMDE_FLOAT64_C(  471.36), SIMDE_FLOAT64_C( -272.46)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  910.81), SIMDE_FLOAT64_C(  911.89)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  439.45), SIMDE_FLOAT64_C( 1184.35)) },
    { simde_x_vloadq_f64(SIMDE_FLOAT64_C( -406.43), SIMDE_FLOAT64_C(  -70.51)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C( -775.79), SIMDE_FLOAT64_C(  949.15)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  369.36), SIMDE_FLOAT64_C( 1019.66)) },
    { simde_x_vloadq_f64(SIMDE_FLOAT64_C(   83.98), SIMDE_FLOAT64_C(  583.58)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  -32.22), SIMDE_FLOAT64_C(  -55.64)),
      simde_x_vloadq_f64(SIMDE_FLOAT64_C(  116.20), SIMDE_FLOAT64_C(  639.22)) },

  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_float64x2_t r = simde_vabdq_f64(test_vec[i].a, test_vec[i].b);
    simde_neon_assert_float64x2_equal(r, test_vec[i].r, 1);
  }

  return MUNIT_OK;
}

HEDLEY_DIAGNOSTIC_PUSH
HEDLEY_DIAGNOSTIC_DISABLE_CAST_QUAL

static MunitTest vabd_tests[] = {
  SIMDE_TESTS_NEON_DEFINE_TEST(s8),
  SIMDE_TESTS_NEON_DEFINE_TEST(s16),
  SIMDE_TESTS_NEON_DEFINE_TEST(s32),
  SIMDE_TESTS_NEON_DEFINE_TEST(u8),
  SIMDE_TESTS_NEON_DEFINE_TEST(u16),
  SIMDE_TESTS_NEON_DEFINE_TEST(u32),
  SIMDE_TESTS_NEON_DEFINE_TEST(f32),
  SIMDE_TESTS_NEON_DEFINE_TEST(f64),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, s8),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, s16),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, s32),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, u8),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, u16),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, u32),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, f32),
  SIMDE_TESTS_NEON_DEFINE_TEST_FULL(q, f64),

  { NULL, NULL, NULL, NULL, MUNIT_TEST_OPTION_NONE, NULL }
};

HEDLEY_C_DECL MunitSuite* SIMDE_TESTS_GENERATE_SYMBOL(SIMDE_TESTS_CURRENT_NEON_OP)(void) {
  static MunitSuite suite = { (char*) "/v" HEDLEY_STRINGIFY(SIMDE_TESTS_CURRENT_NEON_OP), vabd_tests, NULL, 1, MUNIT_SUITE_OPTION_NONE };

  return &suite;
}

HEDLEY_DIAGNOSTIC_POP
