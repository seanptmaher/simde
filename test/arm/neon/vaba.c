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
#include <test/arm/test-arm-internal.h>
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
    { simde_x_vload_s8(INT8_C(-100), INT8_C(  29), INT8_C(  41), INT8_C( -89),
                       INT8_C(  54), INT8_C(  67), INT8_C( 101), INT8_C(   2)),
      simde_x_vload_s8(INT8_C( -86), INT8_C( -46), INT8_C(-114), INT8_C( 121),
                       INT8_C( -83), INT8_C( -68), INT8_C(  99), INT8_C(  -3)),
      simde_x_vload_s8(INT8_C(   4), INT8_C( -68), INT8_C( -75), INT8_C(  11),
                       INT8_C(-127), INT8_C( -31), INT8_C(-124), INT8_C(  20)),
      simde_x_vload_s8(INT8_C(  18), INT8_C(   7), INT8_C(  80), INT8_C( -35),
                       INT8_C(  10), INT8_C( 104), INT8_C(-122), INT8_C(  25)) },
    { simde_x_vload_s8(INT8_C( -95), INT8_C( -84), INT8_C( 116), INT8_C(  62),
                       INT8_C(  86), INT8_C(  26), INT8_C(  76), INT8_C( -99)),
      simde_x_vload_s8(INT8_C(-106), INT8_C( -72), INT8_C(  -3), INT8_C( 101),
                       INT8_C( -10), INT8_C( -84), INT8_C( -14), INT8_C( 121)),
      simde_x_vload_s8(INT8_C( -88), INT8_C(  21), INT8_C( 119), INT8_C(-119),
                       INT8_C( -69), INT8_C(-124), INT8_C( -19), INT8_C(-128)),
      simde_x_vload_s8(INT8_C( -77), INT8_C(  33), INT8_C( -18), INT8_C( -80),
                       INT8_C(  27), INT8_C( -14), INT8_C(  71), INT8_C(  92)) },
    { simde_x_vload_s8(INT8_C( -41), INT8_C(-107), INT8_C( 119), INT8_C( -53),
                       INT8_C( 116), INT8_C(  29), INT8_C(-107), INT8_C( -47)),
      simde_x_vload_s8(INT8_C(  52), INT8_C(   2), INT8_C(  31), INT8_C( -42),
                       INT8_C(  33), INT8_C( -72), INT8_C(  35), INT8_C(   8)),
      simde_x_vload_s8(INT8_C( -50), INT8_C(  83), INT8_C( 113), INT8_C(-123),
                       INT8_C( -98), INT8_C( -11), INT8_C( 115), INT8_C(  43)),
      simde_x_vload_s8(INT8_C(  43), INT8_C( -64), INT8_C( -55), INT8_C(-112),
                       INT8_C( -15), INT8_C(  90), INT8_C(   1), INT8_C(  98)) },
    { simde_x_vload_s8(INT8_C(  24), INT8_C(  69), INT8_C( -22), INT8_C(  -5),
                       INT8_C( 106), INT8_C( -56), INT8_C( 121), INT8_C(   9)),
      simde_x_vload_s8(INT8_C(  14), INT8_C(  91), INT8_C( -97), INT8_C( 115),
                       INT8_C( -84), INT8_C(   8), INT8_C(  -6), INT8_C(  36)),
      simde_x_vload_s8(INT8_C(  15), INT8_C(  76), INT8_C(  52), INT8_C( 107),
                       INT8_C(   8), INT8_C(  19), INT8_C( 107), INT8_C(  -9)),
      simde_x_vload_s8(INT8_C(  25), INT8_C(  98), INT8_C( 127), INT8_C( -29),
                       INT8_C( -58), INT8_C(  83), INT8_C( -22), INT8_C(  18)) },
    { simde_x_vload_s8(INT8_C( -34), INT8_C( 118), INT8_C(  76), INT8_C(-121),
                       INT8_C(  85), INT8_C( 118), INT8_C( -85), INT8_C( 125)),
      simde_x_vload_s8(INT8_C(  48), INT8_C(  61), INT8_C(  48), INT8_C( 126),
                       INT8_C( -61), INT8_C( 122), INT8_C(  53), INT8_C( 116)),
      simde_x_vload_s8(INT8_C(-126), INT8_C( -27), INT8_C(  63), INT8_C( -51),
                       INT8_C( -15), INT8_C(-124), INT8_C(   6), INT8_C(   2)),
      simde_x_vload_s8(INT8_C( -44), INT8_C(  30), INT8_C(  91), INT8_C( -60),
                       INT8_C(-125), INT8_C(-120), INT8_C(-112), INT8_C(  11)) },
    { simde_x_vload_s8(INT8_C( -11), INT8_C(   9), INT8_C(  42), INT8_C( 119),
                       INT8_C( -12), INT8_C(  80), INT8_C( -11), INT8_C( 113)),
      simde_x_vload_s8(INT8_C( 126), INT8_C( 105), INT8_C( -59), INT8_C(  24),
                       INT8_C(  44), INT8_C(-125), INT8_C( -59), INT8_C( -62)),
      simde_x_vload_s8(INT8_C(  39), INT8_C( -93), INT8_C(-117), INT8_C( -71),
                       INT8_C( -16), INT8_C(-115), INT8_C( -41), INT8_C(-110)),
      simde_x_vload_s8(INT8_C( -80), INT8_C(   3), INT8_C( -16), INT8_C(  24),
                       INT8_C(  40), INT8_C(  90), INT8_C(   7), INT8_C(  65)) },
    { simde_x_vload_s8(INT8_C(  79), INT8_C( -34), INT8_C(  67), INT8_C(  12),
                       INT8_C(  99), INT8_C(  16), INT8_C(  79), INT8_C( -34)),
      simde_x_vload_s8(INT8_C( -91), INT8_C(   2), INT8_C(  34), INT8_C( 123),
                       INT8_C(  92), INT8_C(  96), INT8_C( -87), INT8_C( -69)),
      simde_x_vload_s8(INT8_C( -90), INT8_C( -63), INT8_C(  64), INT8_C(-120),
                       INT8_C(  23), INT8_C( -64), INT8_C(-120), INT8_C( -85)),
      simde_x_vload_s8(INT8_C(  80), INT8_C( -27), INT8_C(  97), INT8_C(  -9),
                       INT8_C(  30), INT8_C(  16), INT8_C(  46), INT8_C( -50)) },
    { simde_x_vload_s8(INT8_C(  93), INT8_C(  83), INT8_C( -50), INT8_C( -78),
                       INT8_C(  69), INT8_C( -94), INT8_C(  47), INT8_C( -57)),
      simde_x_vload_s8(INT8_C( -91), INT8_C(-121), INT8_C( -78), INT8_C( 115),
                       INT8_C( -79), INT8_C(  78), INT8_C(  -4), INT8_C( -77)),
      simde_x_vload_s8(INT8_C( -55), INT8_C(  13), INT8_C(-124), INT8_C( -19),
                       INT8_C(  74), INT8_C(-103), INT8_C(  33), INT8_C( -24)),
      simde_x_vload_s8(INT8_C(-127), INT8_C( -39), INT8_C( -96), INT8_C( -82),
                       INT8_C( -34), INT8_C(  69), INT8_C(  84), INT8_C(  -4)) },
  };

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
    { simde_x_vload_s16(INT16_C(  7580), INT16_C(-22743), INT16_C( 17206), INT16_C(   613)),
      simde_x_vload_s16(INT16_C(-11606), INT16_C( 31118), INT16_C(-17235), INT16_C(  -669)),
      simde_x_vload_s16(INT16_C(-17404), INT16_C(  2997), INT16_C( -7807), INT16_C(  5252)),
      simde_x_vload_s16(INT16_C(  1782), INT16_C( -8678), INT16_C( 26634), INT16_C(  6534)) },
    { simde_x_vload_s16(INT16_C(-21343), INT16_C( 15988), INT16_C(  6742), INT16_C(-25268)),
      simde_x_vload_s16(INT16_C(-18282), INT16_C( 26109), INT16_C(-21258), INT16_C( 31218)),
      simde_x_vload_s16(INT16_C(  5544), INT16_C(-30345), INT16_C(-31557), INT16_C(-32531)),
      simde_x_vload_s16(INT16_C(  8605), INT16_C(-20224), INT16_C( -3557), INT16_C( 23955)) },
    { simde_x_vload_s16(INT16_C(-27177), INT16_C(-13449), INT16_C(  7540), INT16_C(-11883)),
      simde_x_vload_s16(INT16_C(   564), INT16_C(-10721), INT16_C(-18399), INT16_C(  2083)),
      simde_x_vload_s16(INT16_C( 21454), INT16_C(-31375), INT16_C( -2658), INT16_C( 11123)),
      simde_x_vload_s16(INT16_C(-16341), INT16_C(-28647), INT16_C( 23281), INT16_C( 25089)) },
    { simde_x_vload_s16(INT16_C( 17688), INT16_C( -1046), INT16_C(-14230), INT16_C(  2425)),
      simde_x_vload_s16(INT16_C( 23310), INT16_C( 29599), INT16_C(  2220), INT16_C(  9466)),
      simde_x_vload_s16(INT16_C( 19471), INT16_C( 27444), INT16_C(  4872), INT16_C( -2197)),
      simde_x_vload_s16(INT16_C( 25093), INT16_C( -7447), INT16_C( 21322), INT16_C(  4844)) },
    { simde_x_vload_s16(INT16_C( 30430), INT16_C(-30900), INT16_C( 30293), INT16_C( 32171)),
      simde_x_vload_s16(INT16_C( 15664), INT16_C( 32304), INT16_C( 31427), INT16_C( 29749)),
      simde_x_vload_s16(INT16_C( -6782), INT16_C(-12993), INT16_C(-31503), INT16_C(   518)),
      simde_x_vload_s16(INT16_C(  7984), INT16_C(-15325), INT16_C(-30369), INT16_C(  2940)) },
    { simde_x_vload_s16(INT16_C(  2549), INT16_C( 30506), INT16_C( 20724), INT16_C( 29173)),
      simde_x_vload_s16(INT16_C( 27006), INT16_C(  6341), INT16_C(-31956), INT16_C(-15675)),
      simde_x_vload_s16(INT16_C(-23769), INT16_C(-18037), INT16_C(-29200), INT16_C(-27945)),
      simde_x_vload_s16(INT16_C(   688), INT16_C(  6128), INT16_C( 23480), INT16_C( 16903)) },
    { simde_x_vload_s16(INT16_C( -8625), INT16_C(  3139), INT16_C(  4195), INT16_C( -8625)),
      simde_x_vload_s16(INT16_C(   677), INT16_C( 31522), INT16_C( 24668), INT16_C(-17495)),
      simde_x_vload_s16(INT16_C(-15962), INT16_C(-30656), INT16_C(-16361), INT16_C(-21624)),
      simde_x_vload_s16(INT16_C( -6660), INT16_C( -2273), INT16_C(  4112), INT16_C(-12754)) },
    { simde_x_vload_s16(INT16_C( 21341), INT16_C(-19762), INT16_C(-23995), INT16_C(-14545)),
      simde_x_vload_s16(INT16_C(-30811), INT16_C( 29618), INT16_C( 20145), INT16_C(-19460)),
      simde_x_vload_s16(INT16_C(  3529), INT16_C( -4732), INT16_C(-26294), INT16_C( -6111)),
      simde_x_vload_s16(INT16_C( -9855), INT16_C(-20888), INT16_C( 17846), INT16_C( -1196)) },
  };

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
    { simde_x_vload_s32(INT32_C(-1490477668), INT32_C(   40190774)),
      simde_x_vload_s32(INT32_C( 2039403178), INT32_C(  -43795283)),
      simde_x_vload_s32(INT32_C(  196459524), INT32_C(  344252801)),
      simde_x_vload_s32(INT32_C(  961545974), INT32_C(  428238858)) },
    { simde_x_vload_s32(INT32_C( 1047833761), INT32_C(-1655956906)),
      simde_x_vload_s32(INT32_C( 1711126678), INT32_C( 2045947126)),
      simde_x_vload_s32(INT32_C(-1988684376), INT32_C(-2131917637)),
      simde_x_vload_s32(INT32_C(-1325391459), INT32_C(-1538854373)) },
    { simde_x_vload_s32(INT32_C( -881355305), INT32_C( -778756748)),
      simde_x_vload_s32(INT32_C( -702610892), INT32_C(  136558625)),
      simde_x_vload_s32(INT32_C(-2056170546), INT32_C(  729019806)),
      simde_x_vload_s32(INT32_C(-1877426133), INT32_C( 1644335179)) },
    { simde_x_vload_s32(INT32_C(  -68532968), INT32_C(  158976106)),
      simde_x_vload_s32(INT32_C( 1939823374), INT32_C(  620365996)),
      simde_x_vload_s32(INT32_C( 1798589455), INT32_C( -143977720)),
      simde_x_vload_s32(INT32_C( -488021499), INT32_C(  317412170)) },
    { simde_x_vload_s32(INT32_C(-2025031970), INT32_C( 2108388949)),
      simde_x_vload_s32(INT32_C( 2117090608), INT32_C( 1949661891)),
      simde_x_vload_s32(INT32_C( -851450494), INT32_C(   33981681)),
      simde_x_vload_s32(INT32_C( -698605776), INT32_C(  192708739)) },
    { simde_x_vload_s32(INT32_C( 1999243765), INT32_C( 1911902452)),
      simde_x_vload_s32(INT32_C(  415590782), INT32_C(-1027243220)),
      simde_x_vload_s32(INT32_C(-1182031065), INT32_C(-1831367184)),
      simde_x_vload_s32(INT32_C(  401621918), INT32_C( -475545560)) },
    { simde_x_vload_s32(INT32_C(  205774415), INT32_C( -565243805)),
      simde_x_vload_s32(INT32_C( 2065826469), INT32_C(-1146527652)),
      simde_x_vload_s32(INT32_C(-2009022042), INT32_C(-1417101289)),
      simde_x_vload_s32(INT32_C( -148969988), INT32_C( -835817442)) },
    { simde_x_vload_s32(INT32_C(-1295101091), INT32_C( -953179579)),
      simde_x_vload_s32(INT32_C( 1941079973), INT32_C(-1275310415)),
      simde_x_vload_s32(INT32_C( -310112823), INT32_C( -400451254)),
      simde_x_vload_s32(INT32_C(  748673409), INT32_C(  -78320418)) },
  };

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
    { simde_x_vload_u8(UINT8_C(156), UINT8_C( 29), UINT8_C( 41), UINT8_C(167),
                       UINT8_C( 54), UINT8_C( 67), UINT8_C(101), UINT8_C(  2)),
      simde_x_vload_u8(UINT8_C(170), UINT8_C(210), UINT8_C(142), UINT8_C(121),
                       UINT8_C(173), UINT8_C(188), UINT8_C( 99), UINT8_C(253)),
      simde_x_vload_u8(UINT8_C(  4), UINT8_C(188), UINT8_C(181), UINT8_C( 11),
                       UINT8_C(129), UINT8_C(225), UINT8_C(132), UINT8_C( 20)),
      simde_x_vload_u8(UINT8_C(246), UINT8_C(  7), UINT8_C( 80), UINT8_C( 57),
                       UINT8_C( 10), UINT8_C(104), UINT8_C(134), UINT8_C( 25)) },
    { simde_x_vload_u8(UINT8_C(161), UINT8_C(172), UINT8_C(116), UINT8_C( 62),
                       UINT8_C( 86), UINT8_C( 26), UINT8_C( 76), UINT8_C(157)),
      simde_x_vload_u8(UINT8_C(150), UINT8_C(184), UINT8_C(253), UINT8_C(101),
                       UINT8_C(246), UINT8_C(172), UINT8_C(242), UINT8_C(121)),
      simde_x_vload_u8(UINT8_C(168), UINT8_C( 21), UINT8_C(119), UINT8_C(137),
                       UINT8_C(187), UINT8_C(132), UINT8_C(237), UINT8_C(128)),
      simde_x_vload_u8(UINT8_C(179), UINT8_C(  9), UINT8_C(238), UINT8_C( 98),
                       UINT8_C( 27), UINT8_C(242), UINT8_C( 71), UINT8_C(164)) },
    { simde_x_vload_u8(UINT8_C(215), UINT8_C(149), UINT8_C(119), UINT8_C(203),
                       UINT8_C(116), UINT8_C( 29), UINT8_C(149), UINT8_C(209)),
      simde_x_vload_u8(UINT8_C( 52), UINT8_C(  2), UINT8_C( 31), UINT8_C(214),
                       UINT8_C( 33), UINT8_C(184), UINT8_C( 35), UINT8_C(  8)),
      simde_x_vload_u8(UINT8_C(206), UINT8_C( 83), UINT8_C(113), UINT8_C(133),
                       UINT8_C(158), UINT8_C(245), UINT8_C(115), UINT8_C( 43)),
      simde_x_vload_u8(UINT8_C(113), UINT8_C(230), UINT8_C(201), UINT8_C(122),
                       UINT8_C(241), UINT8_C( 90), UINT8_C(229), UINT8_C(244)) },
    { simde_x_vload_u8(UINT8_C( 24), UINT8_C( 69), UINT8_C(234), UINT8_C(251),
                       UINT8_C(106), UINT8_C(200), UINT8_C(121), UINT8_C(  9)),
      simde_x_vload_u8(UINT8_C( 14), UINT8_C( 91), UINT8_C(159), UINT8_C(115),
                       UINT8_C(172), UINT8_C(  8), UINT8_C(250), UINT8_C( 36)),
      simde_x_vload_u8(UINT8_C( 15), UINT8_C( 76), UINT8_C( 52), UINT8_C(107),
                       UINT8_C(  8), UINT8_C( 19), UINT8_C(107), UINT8_C(247)),
      simde_x_vload_u8(UINT8_C( 25), UINT8_C( 54), UINT8_C(127), UINT8_C(243),
                       UINT8_C(198), UINT8_C(211), UINT8_C(234), UINT8_C(220)) },
    { simde_x_vload_u8(UINT8_C(222), UINT8_C(118), UINT8_C( 76), UINT8_C(135),
                       UINT8_C( 85), UINT8_C(118), UINT8_C(171), UINT8_C(125)),
      simde_x_vload_u8(UINT8_C( 48), UINT8_C( 61), UINT8_C( 48), UINT8_C(126),
                       UINT8_C(195), UINT8_C(122), UINT8_C( 53), UINT8_C(116)),
      simde_x_vload_u8(UINT8_C(130), UINT8_C(229), UINT8_C( 63), UINT8_C(205),
                       UINT8_C(241), UINT8_C(132), UINT8_C(  6), UINT8_C(  2)),
      simde_x_vload_u8(UINT8_C( 48), UINT8_C( 30), UINT8_C( 91), UINT8_C(214),
                       UINT8_C(131), UINT8_C(128), UINT8_C(124), UINT8_C( 11)) },
    { simde_x_vload_u8(UINT8_C(245), UINT8_C(  9), UINT8_C( 42), UINT8_C(119),
                       UINT8_C(244), UINT8_C( 80), UINT8_C(245), UINT8_C(113)),
      simde_x_vload_u8(UINT8_C(126), UINT8_C(105), UINT8_C(197), UINT8_C( 24),
                       UINT8_C( 44), UINT8_C(131), UINT8_C(197), UINT8_C(194)),
      simde_x_vload_u8(UINT8_C( 39), UINT8_C(163), UINT8_C(139), UINT8_C(185),
                       UINT8_C(240), UINT8_C(141), UINT8_C(215), UINT8_C(146)),
      simde_x_vload_u8(UINT8_C(158), UINT8_C( 67), UINT8_C(240), UINT8_C( 24),
                       UINT8_C(184), UINT8_C( 90), UINT8_C(  7), UINT8_C( 65)) },
    { simde_x_vload_u8(UINT8_C( 79), UINT8_C(222), UINT8_C( 67), UINT8_C( 12),
                       UINT8_C( 99), UINT8_C( 16), UINT8_C( 79), UINT8_C(222)),
      simde_x_vload_u8(UINT8_C(165), UINT8_C(  2), UINT8_C( 34), UINT8_C(123),
                       UINT8_C( 92), UINT8_C( 96), UINT8_C(169), UINT8_C(187)),
      simde_x_vload_u8(UINT8_C(166), UINT8_C(193), UINT8_C( 64), UINT8_C(136),
                       UINT8_C( 23), UINT8_C(192), UINT8_C(136), UINT8_C(171)),
      simde_x_vload_u8(UINT8_C( 80), UINT8_C(157), UINT8_C( 97), UINT8_C( 25),
                       UINT8_C( 30), UINT8_C(112), UINT8_C( 46), UINT8_C(206)) },
    { simde_x_vload_u8(UINT8_C( 93), UINT8_C( 83), UINT8_C(206), UINT8_C(178),
                       UINT8_C( 69), UINT8_C(162), UINT8_C( 47), UINT8_C(199)),
      simde_x_vload_u8(UINT8_C(165), UINT8_C(135), UINT8_C(178), UINT8_C(115),
                       UINT8_C(177), UINT8_C( 78), UINT8_C(252), UINT8_C(179)),
      simde_x_vload_u8(UINT8_C(201), UINT8_C( 13), UINT8_C(132), UINT8_C(237),
                       UINT8_C( 74), UINT8_C(153), UINT8_C( 33), UINT8_C(232)),
      simde_x_vload_u8(UINT8_C(129), UINT8_C(217), UINT8_C(160), UINT8_C( 44),
                       UINT8_C(222), UINT8_C(237), UINT8_C( 84), UINT8_C(252)) },
  };

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
    { simde_x_vload_u16(UINT16_C( 7580), UINT16_C(42793), UINT16_C(17206), UINT16_C(  613)),
      simde_x_vload_u16(UINT16_C(53930), UINT16_C(31118), UINT16_C(48301), UINT16_C(64867)),
      simde_x_vload_u16(UINT16_C(48132), UINT16_C( 2997), UINT16_C(57729), UINT16_C( 5252)),
      simde_x_vload_u16(UINT16_C( 1782), UINT16_C(14672), UINT16_C(26634), UINT16_C( 6534)) },
    { simde_x_vload_u16(UINT16_C(44193), UINT16_C(15988), UINT16_C( 6742), UINT16_C(40268)),
      simde_x_vload_u16(UINT16_C(47254), UINT16_C(26109), UINT16_C(44278), UINT16_C(31218)),
      simde_x_vload_u16(UINT16_C( 5544), UINT16_C(35191), UINT16_C(33979), UINT16_C(33005)),
      simde_x_vload_u16(UINT16_C( 2483), UINT16_C(25070), UINT16_C(61979), UINT16_C(42055)) },
    { simde_x_vload_u16(UINT16_C(38359), UINT16_C(52087), UINT16_C( 7540), UINT16_C(53653)),
      simde_x_vload_u16(UINT16_C(  564), UINT16_C(54815), UINT16_C(47137), UINT16_C( 2083)),
      simde_x_vload_u16(UINT16_C(21454), UINT16_C(34161), UINT16_C(62878), UINT16_C(11123)),
      simde_x_vload_u16(UINT16_C(59249), UINT16_C(31433), UINT16_C(23281), UINT16_C(62693)) },
    { simde_x_vload_u16(UINT16_C(17688), UINT16_C(64490), UINT16_C(51306), UINT16_C( 2425)),
      simde_x_vload_u16(UINT16_C(23310), UINT16_C(29599), UINT16_C( 2220), UINT16_C( 9466)),
      simde_x_vload_u16(UINT16_C(19471), UINT16_C(27444), UINT16_C( 4872), UINT16_C(63339)),
      simde_x_vload_u16(UINT16_C(13849), UINT16_C(62335), UINT16_C(53958), UINT16_C(56298)) },
    { simde_x_vload_u16(UINT16_C(30430), UINT16_C(34636), UINT16_C(30293), UINT16_C(32171)),
      simde_x_vload_u16(UINT16_C(15664), UINT16_C(32304), UINT16_C(31427), UINT16_C(29749)),
      simde_x_vload_u16(UINT16_C(58754), UINT16_C(52543), UINT16_C(34033), UINT16_C(  518)),
      simde_x_vload_u16(UINT16_C( 7984), UINT16_C(54875), UINT16_C(32899), UINT16_C( 2940)) },
    { simde_x_vload_u16(UINT16_C( 2549), UINT16_C(30506), UINT16_C(20724), UINT16_C(29173)),
      simde_x_vload_u16(UINT16_C(27006), UINT16_C( 6341), UINT16_C(33580), UINT16_C(49861)),
      simde_x_vload_u16(UINT16_C(41767), UINT16_C(47499), UINT16_C(36336), UINT16_C(37591)),
      simde_x_vload_u16(UINT16_C(17310), UINT16_C( 6128), UINT16_C(23480), UINT16_C(16903)) },
    { simde_x_vload_u16(UINT16_C(56911), UINT16_C( 3139), UINT16_C( 4195), UINT16_C(56911)),
      simde_x_vload_u16(UINT16_C(  677), UINT16_C(31522), UINT16_C(24668), UINT16_C(48041)),
      simde_x_vload_u16(UINT16_C(49574), UINT16_C(34880), UINT16_C(49175), UINT16_C(43912)),
      simde_x_vload_u16(UINT16_C(40272), UINT16_C( 6497), UINT16_C(28702), UINT16_C(52782)) },
    { simde_x_vload_u16(UINT16_C(21341), UINT16_C(45774), UINT16_C(41541), UINT16_C(50991)),
      simde_x_vload_u16(UINT16_C(34725), UINT16_C(29618), UINT16_C(20145), UINT16_C(46076)),
      simde_x_vload_u16(UINT16_C( 3529), UINT16_C(60804), UINT16_C(39242), UINT16_C(59425)),
      simde_x_vload_u16(UINT16_C(55681), UINT16_C(11424), UINT16_C(60638), UINT16_C(64340)) },
  };

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
    { simde_x_vload_u32(UINT32_C(2804489628), UINT32_C(  40190774)),
      simde_x_vload_u32(UINT32_C(2039403178), UINT32_C(4251172013)),
      simde_x_vload_u32(UINT32_C( 196459524), UINT32_C( 344252801)),
      simde_x_vload_u32(UINT32_C( 961545974), UINT32_C( 428238858)) },
    { simde_x_vload_u32(UINT32_C(1047833761), UINT32_C(2639010390)),
      simde_x_vload_u32(UINT32_C(1711126678), UINT32_C(2045947126)),
      simde_x_vload_u32(UINT32_C(2306282920), UINT32_C(2163049659)),
      simde_x_vload_u32(UINT32_C(1642990003), UINT32_C(2756112923)) },
    { simde_x_vload_u32(UINT32_C(3413611991), UINT32_C(3516210548)),
      simde_x_vload_u32(UINT32_C(3592356404), UINT32_C( 136558625)),
      simde_x_vload_u32(UINT32_C(2238796750), UINT32_C( 729019806)),
      simde_x_vload_u32(UINT32_C(2060052337), UINT32_C(4108671729)) },
    { simde_x_vload_u32(UINT32_C(4226434328), UINT32_C( 158976106)),
      simde_x_vload_u32(UINT32_C(1939823374), UINT32_C( 620365996)),
      simde_x_vload_u32(UINT32_C(1798589455), UINT32_C(4150989576)),
      simde_x_vload_u32(UINT32_C(4085200409), UINT32_C(3689599686)) },
    { simde_x_vload_u32(UINT32_C(2269935326), UINT32_C(2108388949)),
      simde_x_vload_u32(UINT32_C(2117090608), UINT32_C(1949661891)),
      simde_x_vload_u32(UINT32_C(3443516802), UINT32_C(  33981681)),
      simde_x_vload_u32(UINT32_C(3596361520), UINT32_C( 192708739)) },
    { simde_x_vload_u32(UINT32_C(1999243765), UINT32_C(1911902452)),
      simde_x_vload_u32(UINT32_C( 415590782), UINT32_C(3267724076)),
      simde_x_vload_u32(UINT32_C(3112936231), UINT32_C(2463600112)),
      simde_x_vload_u32(UINT32_C( 401621918), UINT32_C(1107778488)) },
    { simde_x_vload_u32(UINT32_C( 205774415), UINT32_C(3729723491)),
      simde_x_vload_u32(UINT32_C(2065826469), UINT32_C(3148439644)),
      simde_x_vload_u32(UINT32_C(2285945254), UINT32_C(2877866007)),
      simde_x_vload_u32(UINT32_C( 425893200), UINT32_C(3459149854)) },
    { simde_x_vload_u32(UINT32_C(2999866205), UINT32_C(3341787717)),
      simde_x_vload_u32(UINT32_C(1941079973), UINT32_C(3019656881)),
      simde_x_vload_u32(UINT32_C(3984854473), UINT32_C(3894516042)),
      simde_x_vload_u32(UINT32_C( 748673409), UINT32_C(4216646878)) },
  };

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
    { simde_x_vloadq_s8(INT8_C(-100), INT8_C(  29), INT8_C(  41), INT8_C( -89),
                        INT8_C(  54), INT8_C(  67), INT8_C( 101), INT8_C(   2),
                        INT8_C( -86), INT8_C( -46), INT8_C(-114), INT8_C( 121),
                        INT8_C( -83), INT8_C( -68), INT8_C(  99), INT8_C(  -3)),
      simde_x_vloadq_s8(INT8_C(   4), INT8_C( -68), INT8_C( -75), INT8_C(  11),
                        INT8_C(-127), INT8_C( -31), INT8_C(-124), INT8_C(  20),
                        INT8_C( -95), INT8_C( -84), INT8_C( 116), INT8_C(  62),
                        INT8_C(  86), INT8_C(  26), INT8_C(  76), INT8_C( -99)),
      simde_x_vloadq_s8(INT8_C(-106), INT8_C( -72), INT8_C(  -3), INT8_C( 101),
                        INT8_C( -10), INT8_C( -84), INT8_C( -14), INT8_C( 121),
                        INT8_C( -88), INT8_C(  21), INT8_C( 119), INT8_C(-119),
                        INT8_C( -69), INT8_C(-124), INT8_C( -19), INT8_C(-128)),
      simde_x_vloadq_s8(INT8_C(  -2), INT8_C(  25), INT8_C( 113), INT8_C( -55),
                        INT8_C( -85), INT8_C(  14), INT8_C( -45), INT8_C(-117),
                        INT8_C( -79), INT8_C(  59), INT8_C(  93), INT8_C( -60),
                        INT8_C( 100), INT8_C( -30), INT8_C(   4), INT8_C( -32)) },
    { simde_x_vloadq_s8(INT8_C( -41), INT8_C(-107), INT8_C( 119), INT8_C( -53),
                        INT8_C( 116), INT8_C(  29), INT8_C(-107), INT8_C( -47),
                        INT8_C(  52), INT8_C(   2), INT8_C(  31), INT8_C( -42),
                        INT8_C(  33), INT8_C( -72), INT8_C(  35), INT8_C(   8)),
      simde_x_vloadq_s8(INT8_C( -50), INT8_C(  83), INT8_C( 113), INT8_C(-123),
                        INT8_C( -98), INT8_C( -11), INT8_C( 115), INT8_C(  43),
                        INT8_C(  24), INT8_C(  69), INT8_C( -22), INT8_C(  -5),
                        INT8_C( 106), INT8_C( -56), INT8_C( 121), INT8_C(   9)),
      simde_x_vloadq_s8(INT8_C(  14), INT8_C(  91), INT8_C( -97), INT8_C( 115),
                        INT8_C( -84), INT8_C(   8), INT8_C(  -6), INT8_C(  36),
                        INT8_C(  15), INT8_C(  76), INT8_C(  52), INT8_C( 107),
                        INT8_C(   8), INT8_C(  19), INT8_C( 107), INT8_C(  -9)),
      simde_x_vloadq_s8(INT8_C(  23), INT8_C(  25), INT8_C( -91), INT8_C( -71),
                        INT8_C(-126), INT8_C(  48), INT8_C( -40), INT8_C( 126),
                        INT8_C(  43), INT8_C(-113), INT8_C( 105), INT8_C(-112),
                        INT8_C(  81), INT8_C(  35), INT8_C( -63), INT8_C(  -8)) },
    { simde_x_vloadq_s8(INT8_C( -34), INT8_C( 118), INT8_C(  76), INT8_C(-121),
                        INT8_C(  85), INT8_C( 118), INT8_C( -85), INT8_C( 125),
                        INT8_C(  48), INT8_C(  61), INT8_C(  48), INT8_C( 126),
                        INT8_C( -61), INT8_C( 122), INT8_C(  53), INT8_C( 116)),
      simde_x_vloadq_s8(INT8_C(-126), INT8_C( -27), INT8_C(  63), INT8_C( -51),
                        INT8_C( -15), INT8_C(-124), INT8_C(   6), INT8_C(   2),
                        INT8_C( -11), INT8_C(   9), INT8_C(  42), INT8_C( 119),
                        INT8_C( -12), INT8_C(  80), INT8_C( -11), INT8_C( 113)),
      simde_x_vloadq_s8(INT8_C( 126), INT8_C( 105), INT8_C( -59), INT8_C(  24),
                        INT8_C(  44), INT8_C(-125), INT8_C( -59), INT8_C( -62),
                        INT8_C(  39), INT8_C( -93), INT8_C(-117), INT8_C( -71),
                        INT8_C( -16), INT8_C(-115), INT8_C( -41), INT8_C(-110)),
      simde_x_vloadq_s8(INT8_C( -38), INT8_C(  -6), INT8_C( -46), INT8_C(  94),
                        INT8_C(-112), INT8_C( 117), INT8_C(  32), INT8_C(  61),
                        INT8_C(  98), INT8_C( -41), INT8_C(-111), INT8_C( -64),
                        INT8_C(  33), INT8_C( -73), INT8_C(  23), INT8_C(-107)) },
    { simde_x_vloadq_s8(INT8_C(  79), INT8_C( -34), INT8_C(  67), INT8_C(  12),
                        INT8_C(  99), INT8_C(  16), INT8_C(  79), INT8_C( -34),
                        INT8_C( -91), INT8_C(   2), INT8_C(  34), INT8_C( 123),
                        INT8_C(  92), INT8_C(  96), INT8_C( -87), INT8_C( -69)),
      simde_x_vloadq_s8(INT8_C( -90), INT8_C( -63), INT8_C(  64), INT8_C(-120),
                        INT8_C(  23), INT8_C( -64), INT8_C(-120), INT8_C( -85),
                        INT8_C(  93), INT8_C(  83), INT8_C( -50), INT8_C( -78),
                        INT8_C(  69), INT8_C( -94), INT8_C(  47), INT8_C( -57)),
      simde_x_vloadq_s8(INT8_C( -91), INT8_C(-121), INT8_C( -78), INT8_C( 115),
                        INT8_C( -79), INT8_C(  78), INT8_C(  -4), INT8_C( -77),
                        INT8_C( -55), INT8_C(  13), INT8_C(-124), INT8_C( -19),
                        INT8_C(  74), INT8_C(-103), INT8_C(  33), INT8_C( -24)),
      simde_x_vloadq_s8(INT8_C(  78), INT8_C( -92), INT8_C( -75), INT8_C(  -9),
                        INT8_C(  -3), INT8_C( -98), INT8_C( -61), INT8_C( -26),
                        INT8_C(-127), INT8_C(  94), INT8_C( -40), INT8_C( -74),
                        INT8_C(  97), INT8_C(  87), INT8_C( -89), INT8_C( -12)) },
    { simde_x_vloadq_s8(INT8_C( 122), INT8_C(  21), INT8_C(  28), INT8_C(-114),
                        INT8_C( -74), INT8_C( 109), INT8_C( -58), INT8_C( -60),
                        INT8_C(  -8), INT8_C(-122), INT8_C( -73), INT8_C( 125),
                        INT8_C(-114), INT8_C(-124), INT8_C( -86), INT8_C(   9)),
      simde_x_vloadq_s8(INT8_C( 118), INT8_C( -29), INT8_C( -28), INT8_C(  -8),
                        INT8_C(   3), INT8_C(   8), INT8_C(  31), INT8_C(-115),
                        INT8_C( -66), INT8_C( -59), INT8_C( -16), INT8_C(  33),
                        INT8_C(  40), INT8_C( -15), INT8_C( -11), INT8_C( 111)),
      simde_x_vloadq_s8(INT8_C( -47), INT8_C(  -2), INT8_C( -46), INT8_C(  84),
                        INT8_C(-116), INT8_C( -36), INT8_C( -97), INT8_C(  14),
                        INT8_C( -73), INT8_C(-104), INT8_C( -56), INT8_C(  96),
                        INT8_C( -64), INT8_C( -48), INT8_C(-126), INT8_C(  -4)),
      simde_x_vloadq_s8(INT8_C( -43), INT8_C(  48), INT8_C(  10), INT8_C( -66),
                        INT8_C( -39), INT8_C(  65), INT8_C(  -8), INT8_C(  69),
                        INT8_C( -15), INT8_C( -41), INT8_C(   1), INT8_C( -68),
                        INT8_C(  90), INT8_C(  61), INT8_C( -51), INT8_C(  98)) },
    { simde_x_vloadq_s8(INT8_C( 109), INT8_C( 125), INT8_C( -51), INT8_C(  -6),
                        INT8_C(  82), INT8_C( -98), INT8_C(  10), INT8_C( -54),
                        INT8_C( -48), INT8_C(  23), INT8_C(  77), INT8_C( 112),
                        INT8_C(  72), INT8_C( 105), INT8_C( -95), INT8_C(   2)),
      simde_x_vloadq_s8(INT8_C(-114), INT8_C(-106), INT8_C( -94), INT8_C( -39),
                        INT8_C( 118), INT8_C( -76), INT8_C( 100), INT8_C( -99),
                        INT8_C(  31), INT8_C(  -2), INT8_C(  -4), INT8_C(   8),
                        INT8_C(  46), INT8_C(  98), INT8_C( -41), INT8_C( -32)),
      simde_x_vloadq_s8(INT8_C(  77), INT8_C( -39), INT8_C(  92), INT8_C(-112),
                        INT8_C(-123), INT8_C(  71), INT8_C(  24), INT8_C( -55),
                        INT8_C( -25), INT8_C( 121), INT8_C( -69), INT8_C(  37),
                        INT8_C( -44), INT8_C(  95), INT8_C(  14), INT8_C( -12)),
      simde_x_vloadq_s8(INT8_C(  44), INT8_C( -64), INT8_C(-121), INT8_C( -79),
                        INT8_C( -87), INT8_C(  93), INT8_C( 114), INT8_C( -10),
                        INT8_C(  54), INT8_C(-110), INT8_C(  12), INT8_C(-115),
                        INT8_C( -18), INT8_C( 102), INT8_C(  68), INT8_C(  22)) },
    { simde_x_vloadq_s8(INT8_C(  51), INT8_C( -97), INT8_C( -65), INT8_C(  84),
                        INT8_C(  44), INT8_C( -56), INT8_C(  81), INT8_C(-127),
                        INT8_C(-126), INT8_C(  91), INT8_C( -54), INT8_C(  51),
                        INT8_C(-113), INT8_C(  -5), INT8_C( 113), INT8_C( -38)),
      simde_x_vloadq_s8(INT8_C(   0), INT8_C( 106), INT8_C( -55), INT8_C( 103),
                        INT8_C(  22), INT8_C( -56), INT8_C( -11), INT8_C(-103),
                        INT8_C( -22), INT8_C( -69), INT8_C(  69), INT8_C( -59),
                        INT8_C( 112), INT8_C(  22), INT8_C(  -4), INT8_C( 121)),
      simde_x_vloadq_s8(INT8_C(  61), INT8_C( -88), INT8_C(  25), INT8_C( 111),
                        INT8_C(-116), INT8_C(  89), INT8_C( 114), INT8_C(  63),
                        INT8_C(  83), INT8_C(  51), INT8_C(   4), INT8_C(  83),
                        INT8_C( -26), INT8_C(  -6), INT8_C( 127), INT8_C(  58)),
      simde_x_vloadq_s8(INT8_C( 112), INT8_C( 115), INT8_C(  35), INT8_C(-126),
                        INT8_C( -94), INT8_C(  89), INT8_C( -50), INT8_C(  87),
                        INT8_C( -69), INT8_C( -45), INT8_C( 127), INT8_C( -63),
                        INT8_C( -57), INT8_C(  21), INT8_C( -12), INT8_C( -39)) },
    { simde_x_vloadq_s8(INT8_C( -55), INT8_C(-123), INT8_C(  27), INT8_C( -32),
                        INT8_C(  52), INT8_C( 126), INT8_C(  -7), INT8_C(-117),
                        INT8_C(-119), INT8_C(-115), INT8_C(  46), INT8_C(  79),
                        INT8_C( -92), INT8_C(  65), INT8_C( -23), INT8_C( -33)),
      simde_x_vloadq_s8(INT8_C( -27), INT8_C(  63), INT8_C(-110), INT8_C( -50),
                        INT8_C(  83), INT8_C(  -4), INT8_C( -70), INT8_C(  47),
                        INT8_C(-107), INT8_C(  24), INT8_C(  69), INT8_C( -92),
                        INT8_C( -59), INT8_C( 113), INT8_C( 116), INT8_C(-104)),
      simde_x_vloadq_s8(INT8_C( -11), INT8_C(  59), INT8_C( -86), INT8_C(  28),
                        INT8_C( 107), INT8_C( -94), INT8_C(  19), INT8_C(  88),
                        INT8_C( -44), INT8_C( -78), INT8_C( -63), INT8_C( -25),
                        INT8_C(  -3), INT8_C(  17), INT8_C(  44), INT8_C( 111)),
      simde_x_vloadq_s8(INT8_C(  17), INT8_C( -11), INT8_C(  51), INT8_C(  46),
                        INT8_C(-118), INT8_C(  36), INT8_C(  82), INT8_C(  -4),
                        INT8_C( -32), INT8_C(  61), INT8_C( -40), INT8_C(-110),
                        INT8_C(  30), INT8_C(  65), INT8_C( -73), INT8_C( -74)) },
  };

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
    { simde_x_vloadq_s16(INT16_C(  7580), INT16_C(-22743), INT16_C( 17206), INT16_C(   613),
                         INT16_C(-11606), INT16_C( 31118), INT16_C(-17235), INT16_C(  -669)),
      simde_x_vloadq_s16(INT16_C(-17404), INT16_C(  2997), INT16_C( -7807), INT16_C(  5252),
                         INT16_C(-21343), INT16_C( 15988), INT16_C(  6742), INT16_C(-25268)),
      simde_x_vloadq_s16(INT16_C(-18282), INT16_C( 26109), INT16_C(-21258), INT16_C( 31218),
                         INT16_C(  5544), INT16_C(-30345), INT16_C(-31557), INT16_C(-32531)),
      simde_x_vloadq_s16(INT16_C(  6702), INT16_C(-13687), INT16_C(  3755), INT16_C(-29679),
                         INT16_C( 15281), INT16_C(-15215), INT16_C( -7580), INT16_C( -7932)) },
    { simde_x_vloadq_s16(INT16_C(-27177), INT16_C(-13449), INT16_C(  7540), INT16_C(-11883),
                         INT16_C(   564), INT16_C(-10721), INT16_C(-18399), INT16_C(  2083)),
      simde_x_vloadq_s16(INT16_C( 21454), INT16_C(-31375), INT16_C( -2658), INT16_C( 11123),
                         INT16_C( 17688), INT16_C( -1046), INT16_C(-14230), INT16_C(  2425)),
      simde_x_vloadq_s16(INT16_C( 23310), INT16_C( 29599), INT16_C(  2220), INT16_C(  9466),
                         INT16_C( 19471), INT16_C( 27444), INT16_C(  4872), INT16_C( -2197)),
      simde_x_vloadq_s16(INT16_C(  6405), INT16_C(-18011), INT16_C( 12418), INT16_C( 32472),
                         INT16_C(-28941), INT16_C(-28417), INT16_C(  9041), INT16_C( -1855)) },
    { simde_x_vloadq_s16(INT16_C( 30430), INT16_C(-30900), INT16_C( 30293), INT16_C( 32171),
                         INT16_C( 15664), INT16_C( 32304), INT16_C( 31427), INT16_C( 29749)),
      simde_x_vloadq_s16(INT16_C( -6782), INT16_C(-12993), INT16_C(-31503), INT16_C(   518),
                         INT16_C(  2549), INT16_C( 30506), INT16_C( 20724), INT16_C( 29173)),
      simde_x_vloadq_s16(INT16_C( 27006), INT16_C(  6341), INT16_C(-31956), INT16_C(-15675),
                         INT16_C(-23769), INT16_C(-18037), INT16_C(-29200), INT16_C(-27945)),
      simde_x_vloadq_s16(INT16_C( -1318), INT16_C( 24248), INT16_C( 29840), INT16_C( 15978),
                         INT16_C(-10654), INT16_C(-16239), INT16_C(-18497), INT16_C(-27369)) },
    { simde_x_vloadq_s16(INT16_C( -8625), INT16_C(  3139), INT16_C(  4195), INT16_C( -8625),
                         INT16_C(   677), INT16_C( 31522), INT16_C( 24668), INT16_C(-17495)),
      simde_x_vloadq_s16(INT16_C(-15962), INT16_C(-30656), INT16_C(-16361), INT16_C(-21624),
                         INT16_C( 21341), INT16_C(-19762), INT16_C(-23995), INT16_C(-14545)),
      simde_x_vloadq_s16(INT16_C(-30811), INT16_C( 29618), INT16_C( 20145), INT16_C(-19460),
                         INT16_C(  3529), INT16_C( -4732), INT16_C(-26294), INT16_C( -6111)),
      simde_x_vloadq_s16(INT16_C(-23474), INT16_C( -2123), INT16_C(-24835), INT16_C( -6461),
                         INT16_C( 24193), INT16_C(-18984), INT16_C( 22369), INT16_C( -3161)) },
    { simde_x_vloadq_s16(INT16_C(  5498), INT16_C(-29156), INT16_C( 28086), INT16_C(-15162),
                         INT16_C(-30984), INT16_C( 32183), INT16_C(-31602), INT16_C(  2474)),
      simde_x_vloadq_s16(INT16_C( -7306), INT16_C( -1820), INT16_C(  2051), INT16_C(-29409),
                         INT16_C(-14914), INT16_C(  8688), INT16_C( -3800), INT16_C( 28661)),
      simde_x_vloadq_s16(INT16_C(  -303), INT16_C( 21714), INT16_C( -9076), INT16_C(  3743),
                         INT16_C(-26441), INT16_C( 24776), INT16_C(-12096), INT16_C(  -894)),
      simde_x_vloadq_s16(INT16_C( 12501), INT16_C(-16486), INT16_C( 16959), INT16_C( 17990),
                         INT16_C(-10371), INT16_C(-17265), INT16_C( 15706), INT16_C( 25293)) },
    { simde_x_vloadq_s16(INT16_C( 32109), INT16_C( -1331), INT16_C(-25006), INT16_C(-13814),
                         INT16_C(  6096), INT16_C( 28749), INT16_C( 26952), INT16_C(   673)),
      simde_x_vloadq_s16(INT16_C(-26994), INT16_C( -9822), INT16_C(-19338), INT16_C(-25244),
                         INT16_C(  -481), INT16_C(  2300), INT16_C( 25134), INT16_C( -7977)),
      simde_x_vloadq_s16(INT16_C( -9907), INT16_C(-28580), INT16_C( 18309), INT16_C(-14056),
                         INT16_C( 31207), INT16_C(  9659), INT16_C( 24532), INT16_C( -3058)),
      simde_x_vloadq_s16(INT16_C(-16340), INT16_C(-20089), INT16_C( 23977), INT16_C( -2626),
                         INT16_C(-27752), INT16_C(-29428), INT16_C( 26350), INT16_C(  5592)) },
    { simde_x_vloadq_s16(INT16_C(-24781), INT16_C( 21695), INT16_C(-14292), INT16_C(-32431),
                         INT16_C( 23426), INT16_C( 13258), INT16_C( -1137), INT16_C( -9615)),
      simde_x_vloadq_s16(INT16_C( 27136), INT16_C( 26569), INT16_C(-14314), INT16_C(-26123),
                         INT16_C(-17430), INT16_C(-15035), INT16_C(  5744), INT16_C( 31228)),
      simde_x_vloadq_s16(INT16_C(-22467), INT16_C( 28441), INT16_C( 22924), INT16_C( 16242),
                         INT16_C( 13139), INT16_C( 21252), INT16_C( -1306), INT16_C( 14975)),
      simde_x_vloadq_s16(INT16_C( 29450), INT16_C(-32221), INT16_C( 22946), INT16_C( 22550),
                         INT16_C(-11541), INT16_C(-15991), INT16_C(  5575), INT16_C( -9718)) },
    { simde_x_vloadq_s16(INT16_C(-31287), INT16_C( -8165), INT16_C( 32308), INT16_C(-29703),
                         INT16_C(-29303), INT16_C( 20270), INT16_C( 16804), INT16_C( -8215)),
      simde_x_vloadq_s16(INT16_C( 16357), INT16_C(-12654), INT16_C(  -941), INT16_C( 12218),
                         INT16_C(  6293), INT16_C(-23483), INT16_C( 29125), INT16_C(-26508)),
      simde_x_vloadq_s16(INT16_C( 15349), INT16_C(  7338), INT16_C(-23957), INT16_C( 22547),
                         INT16_C(-19756), INT16_C( -6207), INT16_C(  4605), INT16_C( 28460)),
      simde_x_vloadq_s16(INT16_C( -2543), INT16_C( 11827), INT16_C(  9292), INT16_C( -1068),
                         INT16_C( 15840), INT16_C(-27990), INT16_C( 16926), INT16_C(-18783)) },
  };

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
    { simde_x_vloadq_s32(INT32_C(-1490477668), INT32_C(   40190774),
                         INT32_C( 2039403178), INT32_C(  -43795283)),
      simde_x_vloadq_s32(INT32_C(  196459524), INT32_C(  344252801),
                         INT32_C( 1047833761), INT32_C(-1655956906)),
      simde_x_vloadq_s32(INT32_C( 1711126678), INT32_C( 2045947126),
                         INT32_C(-1988684376), INT32_C(-2131917637)),
      simde_x_vloadq_s32(INT32_C( -896903426), INT32_C(-1944958143),
                         INT32_C( -997114959), INT32_C( -519756014)) },
    { simde_x_vloadq_s32(INT32_C( -881355305), INT32_C( -778756748),
                         INT32_C( -702610892), INT32_C(  136558625)),
      simde_x_vloadq_s32(INT32_C(-2056170546), INT32_C(  729019806),
                         INT32_C(  -68532968), INT32_C(  158976106)),
      simde_x_vloadq_s32(INT32_C( 1939823374), INT32_C(  620365996),
                         INT32_C( 1798589455), INT32_C( -143977720)),
      simde_x_vloadq_s32(INT32_C(-1180328681), INT32_C( 2128142550),
                         INT32_C(-1862299917), INT32_C( -121560239)) },
    { simde_x_vloadq_s32(INT32_C(-2025031970), INT32_C( 2108388949),
                         INT32_C( 2117090608), INT32_C( 1949661891)),
      simde_x_vloadq_s32(INT32_C( -851450494), INT32_C(   33981681),
                         INT32_C( 1999243765), INT32_C( 1911902452)),
      simde_x_vloadq_s32(INT32_C(  415590782), INT32_C(-1027243220),
                         INT32_C(-1182031065), INT32_C(-1831367184)),
      simde_x_vloadq_s32(INT32_C( 1589172258), INT32_C( 1047164048),
                         INT32_C(-1064184222), INT32_C(-1793607745)) },
    { simde_x_vloadq_s32(INT32_C(  205774415), INT32_C( -565243805),
                         INT32_C( 2065826469), INT32_C(-1146527652)),
      simde_x_vloadq_s32(INT32_C(-2009022042), INT32_C(-1417101289),
                         INT32_C(-1295101091), INT32_C( -953179579)),
      simde_x_vloadq_s32(INT32_C( 1941079973), INT32_C(-1275310415),
                         INT32_C( -310112823), INT32_C( -400451254)),
      simde_x_vloadq_s32(INT32_C( -273716484), INT32_C( -423452931),
                         INT32_C(  623926913), INT32_C( -207103181)) },
    { simde_x_vloadq_s32(INT32_C(-1910762118), INT32_C( -993628746),
                         INT32_C( 2109179640), INT32_C(  162169998)),
      simde_x_vloadq_s32(INT32_C( -119217290), INT32_C(-1927346173),
                         INT32_C(  569427390), INT32_C( 1878389032)),
      simde_x_vloadq_s32(INT32_C( 1423113937), INT32_C(  245357708),
                         INT32_C( 1623759031), INT32_C(  -58535744)),
      simde_x_vloadq_s32(INT32_C(-1080308531), INT32_C( 1179075135),
                         INT32_C(-1131456015), INT32_C( 1657683290)) },
    { simde_x_vloadq_s32(INT32_C(  -87196307), INT32_C( -905273774),
                         INT32_C( 1884100560), INT32_C(   44132680)),
      simde_x_vloadq_s32(INT32_C( -643656050), INT32_C(-1654344586),
                         INT32_C(  150797855), INT32_C( -522755538)),
      simde_x_vloadq_s32(INT32_C(-1872963251), INT32_C( -921155707),
                         INT32_C(  633043431), INT32_C( -200384556)),
      simde_x_vloadq_s32(INT32_C(-1316503508), INT32_C( -172084895),
                         INT32_C(-1928621160), INT32_C(  366503662)) },
    { simde_x_vloadq_s32(INT32_C( 1421844275), INT32_C(-2125346772),
                         INT32_C(  868899714), INT32_C( -630064241)),
      simde_x_vloadq_s32(INT32_C( 1741253120), INT32_C(-1711945706),
                         INT32_C( -985285654), INT32_C( 2046563952)),
      simde_x_vloadq_s32(INT32_C( 1863952445), INT32_C( 1064458636),
                         INT32_C( 1392784211), INT32_C(  981465830)),
      simde_x_vloadq_s32(INT32_C(-2111606006), INT32_C( 1477859702),
                         INT32_C(-1047997717), INT32_C(-1695162363)) },
    { simde_x_vloadq_s32(INT32_C( -535067191), INT32_C(-1946583500),
                         INT32_C( 1328450953), INT32_C( -538361436)),
      simde_x_vloadq_s32(INT32_C( -829276187), INT32_C(  800783443),
                         INT32_C(-1538975595), INT32_C(-1737199163)),
      simde_x_vloadq_s32(INT32_C(  480918517), INT32_C( 1477681771),
                         INT32_C( -406736172), INT32_C( 1865159165)),
      simde_x_vloadq_s32(INT32_C(  775127513), INT32_C(-1269685172),
                         INT32_C( 1020804576), INT32_C(-1230970404)) },
  };

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
    { simde_x_vloadq_u8(UINT8_C(156), UINT8_C( 29), UINT8_C( 41), UINT8_C(167),
                        UINT8_C( 54), UINT8_C( 67), UINT8_C(101), UINT8_C(  2),
                        UINT8_C(170), UINT8_C(210), UINT8_C(142), UINT8_C(121),
                        UINT8_C(173), UINT8_C(188), UINT8_C( 99), UINT8_C(253) ),
      simde_x_vloadq_u8(UINT8_C(  4), UINT8_C(188), UINT8_C(181), UINT8_C( 11),
                        UINT8_C(129), UINT8_C(225), UINT8_C(132), UINT8_C( 20),
                        UINT8_C(161), UINT8_C(172), UINT8_C(116), UINT8_C( 62),
                        UINT8_C( 86), UINT8_C( 26), UINT8_C( 76), UINT8_C(157) ),
      simde_x_vloadq_u8(UINT8_C(150), UINT8_C(184), UINT8_C(253), UINT8_C(101),
                        UINT8_C(246), UINT8_C(172), UINT8_C(242), UINT8_C(121),
                        UINT8_C(168), UINT8_C( 21), UINT8_C(119), UINT8_C(137),
                        UINT8_C(187), UINT8_C(132), UINT8_C(237), UINT8_C(128) ),
      simde_x_vloadq_u8(UINT8_C( 46), UINT8_C( 25), UINT8_C(113), UINT8_C(  1),
                        UINT8_C(171), UINT8_C( 14), UINT8_C(211), UINT8_C(103),
                        UINT8_C(177), UINT8_C( 59), UINT8_C(145), UINT8_C(196),
                        UINT8_C( 18), UINT8_C( 38), UINT8_C(  4), UINT8_C(224) ) },
    { simde_x_vloadq_u8(UINT8_C(215), UINT8_C(149), UINT8_C(119), UINT8_C(203),
                        UINT8_C(116), UINT8_C( 29), UINT8_C(149), UINT8_C(209),
                        UINT8_C( 52), UINT8_C(  2), UINT8_C( 31), UINT8_C(214),
                        UINT8_C( 33), UINT8_C(184), UINT8_C( 35), UINT8_C(  8) ),
      simde_x_vloadq_u8(UINT8_C(206), UINT8_C( 83), UINT8_C(113), UINT8_C(133),
                        UINT8_C(158), UINT8_C(245), UINT8_C(115), UINT8_C( 43),
                        UINT8_C( 24), UINT8_C( 69), UINT8_C(234), UINT8_C(251),
                        UINT8_C(106), UINT8_C(200), UINT8_C(121), UINT8_C(  9) ),
      simde_x_vloadq_u8(UINT8_C( 14), UINT8_C( 91), UINT8_C(159), UINT8_C(115),
                        UINT8_C(172), UINT8_C(  8), UINT8_C(250), UINT8_C( 36),
                        UINT8_C( 15), UINT8_C( 76), UINT8_C( 52), UINT8_C(107),
                        UINT8_C(  8), UINT8_C( 19), UINT8_C(107), UINT8_C(247) ),
      simde_x_vloadq_u8(UINT8_C( 23), UINT8_C(157), UINT8_C(165), UINT8_C(185),
                        UINT8_C(130), UINT8_C( 48), UINT8_C( 28), UINT8_C(202),
                        UINT8_C( 43), UINT8_C(  9), UINT8_C(105), UINT8_C( 70),
                        UINT8_C(191), UINT8_C(  3), UINT8_C( 21), UINT8_C(246) ) },
    { simde_x_vloadq_u8(UINT8_C(222), UINT8_C(118), UINT8_C( 76), UINT8_C(135),
                        UINT8_C( 85), UINT8_C(118), UINT8_C(171), UINT8_C(125),
                        UINT8_C( 48), UINT8_C( 61), UINT8_C( 48), UINT8_C(126),
                        UINT8_C(195), UINT8_C(122), UINT8_C( 53), UINT8_C(116) ),
      simde_x_vloadq_u8(UINT8_C(130), UINT8_C(229), UINT8_C( 63), UINT8_C(205),
                        UINT8_C(241), UINT8_C(132), UINT8_C(  6), UINT8_C(  2),
                        UINT8_C(245), UINT8_C(  9), UINT8_C( 42), UINT8_C(119),
                        UINT8_C(244), UINT8_C( 80), UINT8_C(245), UINT8_C(113) ),
      simde_x_vloadq_u8(UINT8_C(126), UINT8_C(105), UINT8_C(197), UINT8_C( 24),
                        UINT8_C( 44), UINT8_C(131), UINT8_C(197), UINT8_C(194),
                        UINT8_C( 39), UINT8_C(163), UINT8_C(139), UINT8_C(185),
                        UINT8_C(240), UINT8_C(141), UINT8_C(215), UINT8_C(146) ),
      simde_x_vloadq_u8(UINT8_C(218), UINT8_C(250), UINT8_C(210), UINT8_C(210),
                        UINT8_C(144), UINT8_C(117), UINT8_C(106), UINT8_C( 61),
                        UINT8_C( 98), UINT8_C(215), UINT8_C(145), UINT8_C(192),
                        UINT8_C(191), UINT8_C(183), UINT8_C( 23), UINT8_C(149) ) },
    { simde_x_vloadq_u8(UINT8_C( 79), UINT8_C(222), UINT8_C( 67), UINT8_C( 12),
                        UINT8_C( 99), UINT8_C( 16), UINT8_C( 79), UINT8_C(222),
                        UINT8_C(165), UINT8_C(  2), UINT8_C( 34), UINT8_C(123),
                        UINT8_C( 92), UINT8_C( 96), UINT8_C(169), UINT8_C(187) ),
      simde_x_vloadq_u8(UINT8_C(166), UINT8_C(193), UINT8_C( 64), UINT8_C(136),
                        UINT8_C( 23), UINT8_C(192), UINT8_C(136), UINT8_C(171),
                        UINT8_C( 93), UINT8_C( 83), UINT8_C(206), UINT8_C(178),
                        UINT8_C( 69), UINT8_C(162), UINT8_C( 47), UINT8_C(199) ),
      simde_x_vloadq_u8(UINT8_C(165), UINT8_C(135), UINT8_C(178), UINT8_C(115),
                        UINT8_C(177), UINT8_C( 78), UINT8_C(252), UINT8_C(179),
                        UINT8_C(201), UINT8_C( 13), UINT8_C(132), UINT8_C(237),
                        UINT8_C( 74), UINT8_C(153), UINT8_C( 33), UINT8_C(232) ),
      simde_x_vloadq_u8(UINT8_C( 78), UINT8_C(164), UINT8_C(181), UINT8_C(247),
                        UINT8_C(253), UINT8_C(158), UINT8_C(195), UINT8_C(230),
                        UINT8_C( 17), UINT8_C(188), UINT8_C(216), UINT8_C(182),
                        UINT8_C( 97), UINT8_C( 87), UINT8_C(155), UINT8_C(220) ) },
    { simde_x_vloadq_u8(UINT8_C(122), UINT8_C( 21), UINT8_C( 28), UINT8_C(142),
                        UINT8_C(182), UINT8_C(109), UINT8_C(198), UINT8_C(196),
                        UINT8_C(248), UINT8_C(134), UINT8_C(183), UINT8_C(125),
                        UINT8_C(142), UINT8_C(132), UINT8_C(170), UINT8_C(  9) ),
      simde_x_vloadq_u8(UINT8_C(118), UINT8_C(227), UINT8_C(228), UINT8_C(248),
                        UINT8_C(  3), UINT8_C(  8), UINT8_C( 31), UINT8_C(141),
                        UINT8_C(190), UINT8_C(197), UINT8_C(240), UINT8_C( 33),
                        UINT8_C( 40), UINT8_C(241), UINT8_C(245), UINT8_C(111) ),
      simde_x_vloadq_u8(UINT8_C(209), UINT8_C(254), UINT8_C(210), UINT8_C( 84),
                        UINT8_C(140), UINT8_C(220), UINT8_C(159), UINT8_C( 14),
                        UINT8_C(183), UINT8_C(152), UINT8_C(200), UINT8_C( 96),
                        UINT8_C(192), UINT8_C(208), UINT8_C(130), UINT8_C(252) ),
      simde_x_vloadq_u8(UINT8_C(213), UINT8_C( 48), UINT8_C( 10), UINT8_C(234),
                        UINT8_C( 63), UINT8_C( 65), UINT8_C( 70), UINT8_C( 69),
                        UINT8_C(241), UINT8_C( 89), UINT8_C(143), UINT8_C(188),
                        UINT8_C( 38), UINT8_C( 99), UINT8_C( 55), UINT8_C(150) ) },
    { simde_x_vloadq_u8(UINT8_C(109), UINT8_C(125), UINT8_C(205), UINT8_C(250),
                        UINT8_C( 82), UINT8_C(158), UINT8_C( 10), UINT8_C(202),
                        UINT8_C(208), UINT8_C( 23), UINT8_C( 77), UINT8_C(112),
                        UINT8_C( 72), UINT8_C(105), UINT8_C(161), UINT8_C(  2) ),
      simde_x_vloadq_u8(UINT8_C(142), UINT8_C(150), UINT8_C(162), UINT8_C(217),
                        UINT8_C(118), UINT8_C(180), UINT8_C(100), UINT8_C(157),
                        UINT8_C( 31), UINT8_C(254), UINT8_C(252), UINT8_C(  8),
                        UINT8_C( 46), UINT8_C( 98), UINT8_C(215), UINT8_C(224) ),
      simde_x_vloadq_u8(UINT8_C( 77), UINT8_C(217), UINT8_C( 92), UINT8_C(144),
                        UINT8_C(133), UINT8_C( 71), UINT8_C( 24), UINT8_C(201),
                        UINT8_C(231), UINT8_C(121), UINT8_C(187), UINT8_C( 37),
                        UINT8_C(212), UINT8_C( 95), UINT8_C( 14), UINT8_C(244) ),
      simde_x_vloadq_u8(UINT8_C( 44), UINT8_C(192), UINT8_C(135), UINT8_C(177),
                        UINT8_C( 97), UINT8_C( 49), UINT8_C(190), UINT8_C(246),
                        UINT8_C(152), UINT8_C(146), UINT8_C( 12), UINT8_C(141),
                        UINT8_C(238), UINT8_C(102), UINT8_C(216), UINT8_C( 22) ) },
    { simde_x_vloadq_u8(UINT8_C( 51), UINT8_C(159), UINT8_C(191), UINT8_C( 84),
                        UINT8_C( 44), UINT8_C(200), UINT8_C( 81), UINT8_C(129),
                        UINT8_C(130), UINT8_C( 91), UINT8_C(202), UINT8_C( 51),
                        UINT8_C(143), UINT8_C(251), UINT8_C(113), UINT8_C(218) ),
      simde_x_vloadq_u8(UINT8_C(  0), UINT8_C(106), UINT8_C(201), UINT8_C(103),
                        UINT8_C( 22), UINT8_C(200), UINT8_C(245), UINT8_C(153),
                        UINT8_C(234), UINT8_C(187), UINT8_C( 69), UINT8_C(197),
                        UINT8_C(112), UINT8_C( 22), UINT8_C(252), UINT8_C(121) ),
      simde_x_vloadq_u8(UINT8_C( 61), UINT8_C(168), UINT8_C( 25), UINT8_C(111),
                        UINT8_C(140), UINT8_C( 89), UINT8_C(114), UINT8_C( 63),
                        UINT8_C( 83), UINT8_C( 51), UINT8_C(  4), UINT8_C( 83),
                        UINT8_C(230), UINT8_C(250), UINT8_C(127), UINT8_C( 58) ),
      simde_x_vloadq_u8(UINT8_C(112), UINT8_C(221), UINT8_C( 15), UINT8_C( 92),
                        UINT8_C(162), UINT8_C( 89), UINT8_C(206), UINT8_C( 39),
                        UINT8_C(235), UINT8_C(211), UINT8_C(137), UINT8_C(193),
                        UINT8_C(  5), UINT8_C(223), UINT8_C(244), UINT8_C(155) ) },
    { simde_x_vloadq_u8(UINT8_C(201), UINT8_C(133), UINT8_C( 27), UINT8_C(224),
                        UINT8_C( 52), UINT8_C(126), UINT8_C(249), UINT8_C(139),
                        UINT8_C(137), UINT8_C(141), UINT8_C( 46), UINT8_C( 79),
                        UINT8_C(164), UINT8_C( 65), UINT8_C(233), UINT8_C(223) ),
      simde_x_vloadq_u8(UINT8_C(229), UINT8_C( 63), UINT8_C(146), UINT8_C(206),
                        UINT8_C( 83), UINT8_C(252), UINT8_C(186), UINT8_C( 47),
                        UINT8_C(149), UINT8_C( 24), UINT8_C( 69), UINT8_C(164),
                        UINT8_C(197), UINT8_C(113), UINT8_C(116), UINT8_C(152) ),
      simde_x_vloadq_u8(UINT8_C(245), UINT8_C( 59), UINT8_C(170), UINT8_C( 28),
                        UINT8_C(107), UINT8_C(162), UINT8_C( 19), UINT8_C( 88),
                        UINT8_C(212), UINT8_C(178), UINT8_C(193), UINT8_C(231),
                        UINT8_C(253), UINT8_C( 17), UINT8_C( 44), UINT8_C(111) ),
      simde_x_vloadq_u8(UINT8_C(217), UINT8_C(129), UINT8_C( 51), UINT8_C( 46),
                        UINT8_C( 76), UINT8_C( 36), UINT8_C( 82), UINT8_C(180),
                        UINT8_C(200), UINT8_C( 39), UINT8_C(170), UINT8_C(146),
                        UINT8_C(220), UINT8_C(225), UINT8_C(161), UINT8_C(182) ) },
  };

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
    { simde_x_vloadq_u16(UINT16_C( 7580), UINT16_C(42793), UINT16_C(17206), UINT16_C(  613),
                         UINT16_C(53930), UINT16_C(31118), UINT16_C(48301), UINT16_C(64867)),
      simde_x_vloadq_u16(UINT16_C(48132), UINT16_C( 2997), UINT16_C(57729), UINT16_C( 5252),
                         UINT16_C(44193), UINT16_C(15988), UINT16_C( 6742), UINT16_C(40268)),
      simde_x_vloadq_u16(UINT16_C(47254), UINT16_C(26109), UINT16_C(44278), UINT16_C(31218),
                         UINT16_C( 5544), UINT16_C(35191), UINT16_C(33979), UINT16_C(33005)),
      simde_x_vloadq_u16(UINT16_C( 6702), UINT16_C(  369), UINT16_C( 3755), UINT16_C(26579),
                         UINT16_C(15281), UINT16_C(50321), UINT16_C(10002), UINT16_C(57604)) },
    { simde_x_vloadq_u16(UINT16_C(38359), UINT16_C(52087), UINT16_C( 7540), UINT16_C(53653),
                         UINT16_C(  564), UINT16_C(54815), UINT16_C(47137), UINT16_C( 2083)),
      simde_x_vloadq_u16(UINT16_C(21454), UINT16_C(34161), UINT16_C(62878), UINT16_C(11123),
                         UINT16_C(17688), UINT16_C(64490), UINT16_C(51306), UINT16_C( 2425)),
      simde_x_vloadq_u16(UINT16_C(23310), UINT16_C(29599), UINT16_C( 2220), UINT16_C( 9466),
                         UINT16_C(19471), UINT16_C(27444), UINT16_C( 4872), UINT16_C(63339)),
      simde_x_vloadq_u16(UINT16_C(40215), UINT16_C(47525), UINT16_C(12418), UINT16_C(51996),
                         UINT16_C( 2347), UINT16_C(17769), UINT16_C(  703), UINT16_C(62997)) },
    { simde_x_vloadq_u16(UINT16_C(30430), UINT16_C(34636), UINT16_C(30293), UINT16_C(32171),
                         UINT16_C(15664), UINT16_C(32304), UINT16_C(31427), UINT16_C(29749)),
      simde_x_vloadq_u16(UINT16_C(58754), UINT16_C(52543), UINT16_C(34033), UINT16_C(  518),
                         UINT16_C( 2549), UINT16_C(30506), UINT16_C(20724), UINT16_C(29173)),
      simde_x_vloadq_u16(UINT16_C(27006), UINT16_C( 6341), UINT16_C(33580), UINT16_C(49861),
                         UINT16_C(41767), UINT16_C(47499), UINT16_C(36336), UINT16_C(37591)),
      simde_x_vloadq_u16(UINT16_C(64218), UINT16_C(53970), UINT16_C(29840), UINT16_C(15978),
                         UINT16_C(54882), UINT16_C(49297), UINT16_C(47039), UINT16_C(38167)) },
    { simde_x_vloadq_u16(UINT16_C(56911), UINT16_C( 3139), UINT16_C( 4195), UINT16_C(56911),
                         UINT16_C(  677), UINT16_C(31522), UINT16_C(24668), UINT16_C(48041)),
      simde_x_vloadq_u16(UINT16_C(49574), UINT16_C(34880), UINT16_C(49175), UINT16_C(43912),
                         UINT16_C(21341), UINT16_C(45774), UINT16_C(41541), UINT16_C(50991)),
      simde_x_vloadq_u16(UINT16_C(34725), UINT16_C(29618), UINT16_C(20145), UINT16_C(46076),
                         UINT16_C( 3529), UINT16_C(60804), UINT16_C(39242), UINT16_C(59425)),
      simde_x_vloadq_u16(UINT16_C(42062), UINT16_C(63413), UINT16_C(40701), UINT16_C(59075),
                         UINT16_C(48401), UINT16_C(46552), UINT16_C(22369), UINT16_C(56475)) },
    { simde_x_vloadq_u16(UINT16_C( 5498), UINT16_C(36380), UINT16_C(28086), UINT16_C(50374),
                         UINT16_C(34552), UINT16_C(32183), UINT16_C(33934), UINT16_C( 2474)),
      simde_x_vloadq_u16(UINT16_C(58230), UINT16_C(63716), UINT16_C( 2051), UINT16_C(36127),
                         UINT16_C(50622), UINT16_C( 8688), UINT16_C(61736), UINT16_C(28661)),
      simde_x_vloadq_u16(UINT16_C(65233), UINT16_C(21714), UINT16_C(56460), UINT16_C( 3743),
                         UINT16_C(39095), UINT16_C(24776), UINT16_C(53440), UINT16_C(64642)),
      simde_x_vloadq_u16(UINT16_C(12501), UINT16_C(59914), UINT16_C(16959), UINT16_C(17990),
                         UINT16_C(23025), UINT16_C(48271), UINT16_C(25638), UINT16_C(38455)) },
    { simde_x_vloadq_u16(UINT16_C(32109), UINT16_C(64205), UINT16_C(40530), UINT16_C(51722),
                         UINT16_C( 6096), UINT16_C(28749), UINT16_C(26952), UINT16_C(  673)),
      simde_x_vloadq_u16(UINT16_C(38542), UINT16_C(55714), UINT16_C(46198), UINT16_C(40292),
                         UINT16_C(65055), UINT16_C( 2300), UINT16_C(25134), UINT16_C(57559)),
      simde_x_vloadq_u16(UINT16_C(55629), UINT16_C(36956), UINT16_C(18309), UINT16_C(51480),
                         UINT16_C(31207), UINT16_C( 9659), UINT16_C(24532), UINT16_C(62478)),
      simde_x_vloadq_u16(UINT16_C(49196), UINT16_C(45447), UINT16_C(12641), UINT16_C(62910),
                         UINT16_C(37784), UINT16_C(36108), UINT16_C(26350), UINT16_C( 5592)) },
    { simde_x_vloadq_u16(UINT16_C(40755), UINT16_C(21695), UINT16_C(51244), UINT16_C(33105),
                         UINT16_C(23426), UINT16_C(13258), UINT16_C(64399), UINT16_C(55921)),
      simde_x_vloadq_u16(UINT16_C(27136), UINT16_C(26569), UINT16_C(51222), UINT16_C(39413),
                         UINT16_C(48106), UINT16_C(50501), UINT16_C( 5744), UINT16_C(31228)),
      simde_x_vloadq_u16(UINT16_C(43069), UINT16_C(28441), UINT16_C(22924), UINT16_C(16242),
                         UINT16_C(13139), UINT16_C(21252), UINT16_C(64230), UINT16_C(14975)),
      simde_x_vloadq_u16(UINT16_C(56688), UINT16_C(23567), UINT16_C(22946), UINT16_C( 9934),
                         UINT16_C(53995), UINT16_C(49545), UINT16_C(57349), UINT16_C(39668)) },
    { simde_x_vloadq_u16(UINT16_C(34249), UINT16_C(57371), UINT16_C(32308), UINT16_C(35833),
                         UINT16_C(36233), UINT16_C(20270), UINT16_C(16804), UINT16_C(57321)),
      simde_x_vloadq_u16(UINT16_C(16357), UINT16_C(52882), UINT16_C(64595), UINT16_C(12218),
                         UINT16_C( 6293), UINT16_C(42053), UINT16_C(29125), UINT16_C(39028)),
      simde_x_vloadq_u16(UINT16_C(15349), UINT16_C( 7338), UINT16_C(41579), UINT16_C(22547),
                         UINT16_C(45780), UINT16_C(59329), UINT16_C( 4605), UINT16_C(28460)),
      simde_x_vloadq_u16(UINT16_C(33241), UINT16_C(11827), UINT16_C( 9292), UINT16_C(46162),
                         UINT16_C(10184), UINT16_C(37546), UINT16_C(57820), UINT16_C(46753)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint16x8_t r = simde_vabaq_u16(test_vec[i].a, test_vec[i].b, test_vec[i].c);
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
    { simde_x_vloadq_u32(UINT32_C(2804489628), UINT32_C(  40190774),
                         UINT32_C(2039403178), UINT32_C(4251172013)),
      simde_x_vloadq_u32(UINT32_C( 196459524), UINT32_C( 344252801),
                         UINT32_C(1047833761), UINT32_C(2639010390)),
      simde_x_vloadq_u32(UINT32_C(1711126678), UINT32_C(2045947126),
                         UINT32_C(2306282920), UINT32_C(2163049659)),
      simde_x_vloadq_u32(UINT32_C(  24189486), UINT32_C(1741885099),
                         UINT32_C(3297852337), UINT32_C(3775211282)) },
    { simde_x_vloadq_u32(UINT32_C(3413611991), UINT32_C(3516210548),
                         UINT32_C(3592356404), UINT32_C( 136558625)),
      simde_x_vloadq_u32(UINT32_C(2238796750), UINT32_C( 729019806),
                         UINT32_C(4226434328), UINT32_C( 158976106)),
      simde_x_vloadq_u32(UINT32_C(1939823374), UINT32_C( 620365996),
                         UINT32_C(1798589455), UINT32_C(4150989576)),
      simde_x_vloadq_u32(UINT32_C(3114638615), UINT32_C(3407556738),
                         UINT32_C(1164511531), UINT32_C(4128572095)) },
    { simde_x_vloadq_u32(UINT32_C(2269935326), UINT32_C(2108388949),
                         UINT32_C(2117090608), UINT32_C(1949661891)),
      simde_x_vloadq_u32(UINT32_C(3443516802), UINT32_C(  33981681),
                         UINT32_C(1999243765), UINT32_C(1911902452)),
      simde_x_vloadq_u32(UINT32_C( 415590782), UINT32_C(3267724076),
                         UINT32_C(3112936231), UINT32_C(2463600112)),
      simde_x_vloadq_u32(UINT32_C(3536976602), UINT32_C(1047164048),
                         UINT32_C(3230783074), UINT32_C(2501359551)) },
    { simde_x_vloadq_u32(UINT32_C( 205774415), UINT32_C(3729723491),
                         UINT32_C(2065826469), UINT32_C(3148439644)),
      simde_x_vloadq_u32(UINT32_C(2285945254), UINT32_C(2877866007),
                         UINT32_C(2999866205), UINT32_C(3341787717)),
      simde_x_vloadq_u32(UINT32_C(1941079973), UINT32_C(3019656881),
                         UINT32_C(3984854473), UINT32_C(3894516042)),
      simde_x_vloadq_u32(UINT32_C(4155876430), UINT32_C(3871514365),
                         UINT32_C(3050814737), UINT32_C(3701167969)) },
    { simde_x_vloadq_u32(UINT32_C(2384205178), UINT32_C(3301338550),
                         UINT32_C(2109179640), UINT32_C( 162169998)),
      simde_x_vloadq_u32(UINT32_C(4175750006), UINT32_C(2367621123),
                         UINT32_C( 569427390), UINT32_C(1878389032)),
      simde_x_vloadq_u32(UINT32_C(1423113937), UINT32_C( 245357708),
                         UINT32_C(1623759031), UINT32_C(4236431552)),
      simde_x_vloadq_u32(UINT32_C(3926536405), UINT32_C(1179075135),
                         UINT32_C(3163511281), UINT32_C(2520212518)) },
    { simde_x_vloadq_u32(UINT32_C(4207770989), UINT32_C(3389693522),
                         UINT32_C(1884100560), UINT32_C(  44132680)),
      simde_x_vloadq_u32(UINT32_C(3651311246), UINT32_C(2640622710),
                         UINT32_C( 150797855), UINT32_C(3772211758)),
      simde_x_vloadq_u32(UINT32_C(2422004045), UINT32_C(3373811589),
                         UINT32_C( 633043431), UINT32_C(4094582740)),
      simde_x_vloadq_u32(UINT32_C(2978463788), UINT32_C(4122882401),
                         UINT32_C(2366346136), UINT32_C( 366503662)) },
    { simde_x_vloadq_u32(UINT32_C(1421844275), UINT32_C(2169620524),
                         UINT32_C( 868899714), UINT32_C(3664903055)),
      simde_x_vloadq_u32(UINT32_C(1741253120), UINT32_C(2583021590),
                         UINT32_C(3309681642), UINT32_C(2046563952)),
      simde_x_vloadq_u32(UINT32_C(1863952445), UINT32_C(1064458636),
                         UINT32_C(1392784211), UINT32_C( 981465830)),
      simde_x_vloadq_u32(UINT32_C(1544543600), UINT32_C( 651057570),
                         UINT32_C(3246969579), UINT32_C(2599804933)) },
    { simde_x_vloadq_u32(UINT32_C(3759900105), UINT32_C(2348383796),
                         UINT32_C(1328450953), UINT32_C(3756605860)),
      simde_x_vloadq_u32(UINT32_C(3465691109), UINT32_C( 800783443),
                         UINT32_C(2755991701), UINT32_C(2557768133)),
      simde_x_vloadq_u32(UINT32_C( 480918517), UINT32_C(1477681771),
                         UINT32_C(3888231124), UINT32_C(1865159165)),
      simde_x_vloadq_u32(UINT32_C( 775127513), UINT32_C(3025282124),
                         UINT32_C(2460690376), UINT32_C(3063996892)) },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    simde_uint32x4_t r = simde_vabaq_u32(test_vec[i].a, test_vec[i].b, test_vec[i].c);
    simde_neon_assert_uint32x4(r, ==, test_vec[i].r);
  }

  return MUNIT_OK;
}

static MunitTest vaba_tests[] = {
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
  static MunitSuite suite = { (char*) "/v" HEDLEY_STRINGIFY(SIMDE_TESTS_CURRENT_NEON_OP), vaba_tests, NULL, 1, MUNIT_SUITE_OPTION_NONE };
  return &suite;
};
