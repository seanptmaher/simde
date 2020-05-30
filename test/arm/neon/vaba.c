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
    { simde_x_vload_s8(INT8_C( -48), INT8_C(-128), INT8_C(  91), INT8_C(  96),
                       INT8_C(-115), INT8_C(  59), INT8_C(  49), INT8_C(  27)),
      simde_x_vload_s8(INT8_C(   1), INT8_C(-110), INT8_C(  40), INT8_C(-127),
                       INT8_C(  31), INT8_C(  -5), INT8_C( -43), INT8_C(  40)),
      simde_x_vload_s8(INT8_C( -93), INT8_C(-107), INT8_C( 103), INT8_C( -67),
                       INT8_C(  52), INT8_C(-106), INT8_C(  23), INT8_C( -52)),
      simde_x_vload_s8(INT8_C(  46), INT8_C(-125), INT8_C(-102), INT8_C(-100),
                       INT8_C( -94), INT8_C( -96), INT8_C( 115), INT8_C( 119)) },
    { simde_x_vload_s8(INT8_C(  23), INT8_C(-100), INT8_C(-108), INT8_C( 118),
                       INT8_C( -68), INT8_C(   4), INT8_C(  86), INT8_C( -96)),
      simde_x_vload_s8(INT8_C( -95), INT8_C(-122), INT8_C( 107), INT8_C( -27),
                       INT8_C(  25), INT8_C(  52), INT8_C(-101), INT8_C(  87)),
      simde_x_vload_s8(INT8_C( -89), INT8_C(  47), INT8_C(  36), INT8_C( -10),
                       INT8_C( 113), INT8_C(  32), INT8_C( 119), INT8_C( -60)),
      simde_x_vload_s8(INT8_C(  29), INT8_C(  69), INT8_C( -37), INT8_C(-121),
                       INT8_C(  20), INT8_C(  24), INT8_C(  50), INT8_C(  51)) },
    { simde_x_vload_s8(INT8_C(  80), INT8_C(  -2), INT8_C( -37), INT8_C(  55),
                       INT8_C(-111), INT8_C(  35), INT8_C(  91), INT8_C( -92)),
      simde_x_vload_s8(INT8_C( -93), INT8_C( -88), INT8_C(  74), INT8_C( -28),
                       INT8_C( -61), INT8_C(  16), INT8_C( -35), INT8_C(  49)),
      simde_x_vload_s8(INT8_C( -30), INT8_C(  22), INT8_C( 115), INT8_C(  -1),
                       INT8_C( -57), INT8_C( -18), INT8_C(-121), INT8_C( -67)),
      simde_x_vload_s8(INT8_C(-113), INT8_C( 108), INT8_C(   4), INT8_C(  82),
                       INT8_C(-107), INT8_C(  69), INT8_C( -79), INT8_C(  24)) },
    { simde_x_vload_s8(INT8_C( -89), INT8_C( 125), INT8_C( -90), INT8_C(  37),
                       INT8_C(  32), INT8_C(  55), INT8_C(  47), INT8_C(  80)),
      simde_x_vload_s8(INT8_C( -12), INT8_C( -24), INT8_C(-119), INT8_C(  31),
                       INT8_C(  46), INT8_C(  91), INT8_C(  33), INT8_C( -45)),
      simde_x_vload_s8(INT8_C(  23), INT8_C(  17), INT8_C( -75), INT8_C(  53),
                       INT8_C(-116), INT8_C(  58), INT8_C( -88), INT8_C(  19)),
      simde_x_vload_s8(INT8_C( -54), INT8_C( -90), INT8_C( -46), INT8_C(  59),
                       INT8_C( -62), INT8_C(  88), INT8_C( -88), INT8_C(-112)) },
    { simde_x_vload_s8(INT8_C(  -2), INT8_C(  46), INT8_C(-113), INT8_C(  -9),
                       INT8_C(   5), INT8_C( -44), INT8_C( -35), INT8_C(-125)),
      simde_x_vload_s8(INT8_C(  40), INT8_C(-100), INT8_C(  80), INT8_C( -69),
                       INT8_C(  10), INT8_C(  30), INT8_C(  17), INT8_C(  96)),
      simde_x_vload_s8(INT8_C(  10), INT8_C( -60), INT8_C(-102), INT8_C(  88),
                       INT8_C( 102), INT8_C( -32), INT8_C( 127), INT8_C( -64)),
      simde_x_vload_s8(INT8_C(  28), INT8_C(  86), INT8_C(  69), INT8_C(-108),
                       INT8_C(  97), INT8_C(  18), INT8_C(  75), INT8_C(  35)) },
    { simde_x_vload_s8(INT8_C(  89), INT8_C(  63), INT8_C( -92), INT8_C( -17),
                       INT8_C( -18), INT8_C( 116), INT8_C( -69), INT8_C(  30)),
      simde_x_vload_s8(INT8_C( -34), INT8_C( -83), INT8_C( -75), INT8_C(  57),
                       INT8_C(-114), INT8_C( -57), INT8_C(  17), INT8_C(  90)),
      simde_x_vload_s8(INT8_C(-122), INT8_C( 123), INT8_C(  61), INT8_C( 108),
                       INT8_C( 114), INT8_C(-125), INT8_C(-102), INT8_C(-100)),
      simde_x_vload_s8(INT8_C( -79), INT8_C(  13), INT8_C(  44), INT8_C(  34),
                       INT8_C( -46), INT8_C( -72), INT8_C(  50), INT8_C( -36)) },
    { simde_x_vload_s8(INT8_C( -43), INT8_C(  48), INT8_C(  81), INT8_C(-104),
                       INT8_C( 123), INT8_C(-117), INT8_C( -84), INT8_C( -76)),
      simde_x_vload_s8(INT8_C( -23), INT8_C(  25), INT8_C( -44), INT8_C( -10),
                       INT8_C( 106), INT8_C( -34), INT8_C(  90), INT8_C(  32)),
      simde_x_vload_s8(INT8_C(   2), INT8_C( -24), INT8_C( -64), INT8_C( -68),
                       INT8_C(-109), INT8_C( -64), INT8_C(-126), INT8_C( 116)),
      simde_x_vload_s8(INT8_C( -18), INT8_C(  97), INT8_C( 101), INT8_C( -46),
                       INT8_C(  82), INT8_C( -87), INT8_C(-124), INT8_C(   8)) },
    { simde_x_vload_s8(INT8_C( -10), INT8_C( -13), INT8_C( -43), INT8_C( -92),
                       INT8_C(  88), INT8_C( -85), INT8_C( 111), INT8_C( -38)),
      simde_x_vload_s8(INT8_C( -54), INT8_C(  90), INT8_C( -15), INT8_C( -62),
                       INT8_C( -63), INT8_C(  24), INT8_C( -42), INT8_C( -55)),
      simde_x_vload_s8(INT8_C( -62), INT8_C( -84), INT8_C( -51), INT8_C(-124),
                       INT8_C(  44), INT8_C( -18), INT8_C( 107), INT8_C(  72)),
      simde_x_vload_s8(INT8_C(  -2), INT8_C( -95), INT8_C(  -7), INT8_C( -30),
                       INT8_C( -61), INT8_C( -43), INT8_C(   4), INT8_C(  89)) },
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
    { simde_x_vload_s16(INT16_C(-32560), INT16_C( 24667), INT16_C( 15245), INT16_C(  6961)),
      simde_x_vload_s16(INT16_C(-28159), INT16_C(-32472), INT16_C( -1249), INT16_C( 10453)),
      simde_x_vload_s16(INT16_C(-27229), INT16_C(-17049), INT16_C(-27084), INT16_C(-13289)),
      simde_x_vload_s16(INT16_C(-31630), INT16_C(-25446), INT16_C(-24456), INT16_C( 30703)) },
    { simde_x_vload_s16(INT16_C(-25577), INT16_C( 30356), INT16_C(  1212), INT16_C(-24490)),
      simde_x_vload_s16(INT16_C(-31071), INT16_C( -6805), INT16_C( 13337), INT16_C( 22427)),
      simde_x_vload_s16(INT16_C( 12199), INT16_C( -2524), INT16_C(  8305), INT16_C(-15241)),
      simde_x_vload_s16(INT16_C( 17693), INT16_C(-30899), INT16_C(  6244), INT16_C( 13178)) },
    { simde_x_vload_s16(INT16_C(  -432), INT16_C( 14299), INT16_C(  9105), INT16_C(-23461)),
      simde_x_vload_s16(INT16_C(-22365), INT16_C( -7094), INT16_C(  4291), INT16_C( 12765)),
      simde_x_vload_s16(INT16_C(  5858), INT16_C(  -141), INT16_C( -4409), INT16_C(-17017)),
      simde_x_vload_s16(INT16_C( 27791), INT16_C( 21252), INT16_C( 17805), INT16_C(  6321)) },
    { simde_x_vload_s16(INT16_C( 32167), INT16_C(  9638), INT16_C( 14112), INT16_C( 20527)),
      simde_x_vload_s16(INT16_C( -5900), INT16_C(  8073), INT16_C( 23342), INT16_C(-11487)),
      simde_x_vload_s16(INT16_C(  4375), INT16_C( 13749), INT16_C( 14988), INT16_C(  5032)),
      simde_x_vload_s16(INT16_C(-23094), INT16_C( 15314), INT16_C( 22466), INT16_C(-28490)) },
    { simde_x_vload_s16(INT16_C( 12030), INT16_C( -2161), INT16_C(-11259), INT16_C(-31779)),
      simde_x_vload_s16(INT16_C(-25560), INT16_C(-17584), INT16_C(  7690), INT16_C( 24593)),
      simde_x_vload_s16(INT16_C(-15350), INT16_C( 22682), INT16_C( -8090), INT16_C(-16257)),
      simde_x_vload_s16(INT16_C( 22240), INT16_C(-27431), INT16_C(  4521), INT16_C(  9071)) },
    { simde_x_vload_s16(INT16_C( 16217), INT16_C( -4188), INT16_C( 29934), INT16_C(  7867)),
      simde_x_vload_s16(INT16_C(-21026), INT16_C( 14773), INT16_C(-14450), INT16_C( 23057)),
      simde_x_vload_s16(INT16_C( 31622), INT16_C( 27709), INT16_C(-31886), INT16_C(-25446)),
      simde_x_vload_s16(INT16_C(  3329), INT16_C(  8748), INT16_C(-18166), INT16_C( -9166)) },
    { simde_x_vload_s16(INT16_C( 12501), INT16_C(-26543), INT16_C(-29829), INT16_C(-19284)),
      simde_x_vload_s16(INT16_C(  6633), INT16_C( -2348), INT16_C( -8598), INT16_C(  8282)),
      simde_x_vload_s16(INT16_C( -6142), INT16_C(-17216), INT16_C(-16237), INT16_C( 29826)),
      simde_x_vload_s16(INT16_C( 25276), INT16_C(-11675), INT16_C(-22190), INT16_C(  2260)) },
    { simde_x_vload_s16(INT16_C( -3082), INT16_C(-23339), INT16_C(-21672), INT16_C( -9617)),
      simde_x_vload_s16(INT16_C( 23242), INT16_C(-15631), INT16_C(  6337), INT16_C(-13866)),
      simde_x_vload_s16(INT16_C(-21310), INT16_C(-31539), INT16_C( -4564), INT16_C( 18539)),
      simde_x_vload_s16(INT16_C(-24066), INT16_C( -7431), INT16_C(-10771), INT16_C( 22788)) },
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
    { simde_x_vload_s32(INT32_C( 1616609488), INT32_C(  456211341)),
      simde_x_vload_s32(INT32_C(-2128047615), INT32_C(  685112095)),
      simde_x_vload_s32(INT32_C(-1117284957), INT32_C( -870869452)),
      simde_x_vload_s32(INT32_C(-1667595150), INT32_C( 2012192888)) },
    { simde_x_vload_s32(INT32_C( 1989450775), INT32_C(-1604975428)),
      simde_x_vload_s32(INT32_C( -445938015), INT32_C( 1469789209)),
      simde_x_vload_s32(INT32_C( -165400665), INT32_C( -998825871)),
      simde_x_vload_s32(INT32_C(-2024979171), INT32_C(  221376788)) },
    { simde_x_vload_s32(INT32_C(  937164368), INT32_C(-1537530991)),
      simde_x_vload_s32(INT32_C( -464869213), INT32_C(  836571331)),
      simde_x_vload_s32(INT32_C(   -9234718), INT32_C(-1115164985)),
      simde_x_vload_s32(INT32_C( 1392798863), INT32_C(  414205325)) },
    { simde_x_vload_s32(INT32_C(  631668135), INT32_C( 1345271584)),
      simde_x_vload_s32(INT32_C(  529131764), INT32_C( -752788690)),
      simde_x_vload_s32(INT32_C(  901058839), INT32_C(  329792140)),
      simde_x_vload_s32(INT32_C( 1003595210), INT32_C(-1867114882)) },
    { simde_x_vload_s32(INT32_C( -141611266), INT32_C(-2082614267)),
      simde_x_vload_s32(INT32_C(-1152345048), INT32_C( 1611734538)),
      simde_x_vload_s32(INT32_C( 1486537738), INT32_C(-1065361306)),
      simde_x_vload_s32(INT32_C( 1514473244), INT32_C( -464742815)) },
    { simde_x_vload_s32(INT32_C( -274448551), INT32_C(  515601646)),
      simde_x_vload_s32(INT32_C(  968207838), INT32_C( 1511114638)),
      simde_x_vload_s32(INT32_C( 1815968646), INT32_C(-1667595406)),
      simde_x_vload_s32(INT32_C(  573312257), INT32_C( 1631858898)) },
    { simde_x_vload_s32(INT32_C(-1739509547), INT32_C(-1263760517)),
      simde_x_vload_s32(INT32_C( -153871895), INT32_C(  542826090)),
      simde_x_vload_s32(INT32_C(-1128208382), INT32_C( 1954726035)),
      simde_x_vload_s32(INT32_C( -765173060), INT32_C(  148139428)) },
    { simde_x_vload_s32(INT32_C(-1529482250), INT32_C( -630215848)),
      simde_x_vload_s32(INT32_C(-1024369974), INT32_C( -908715839)),
      simde_x_vload_s32(INT32_C(-2066895678), INT32_C( 1215032876)),
      simde_x_vload_s32(INT32_C( -486956546), INT32_C( 1493532867)) },
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
    { simde_x_vload_u8(UINT8_C(208), UINT8_C(128), UINT8_C( 91), UINT8_C( 96),
                       UINT8_C(141), UINT8_C( 59), UINT8_C( 49), UINT8_C( 27)),
      simde_x_vload_u8(UINT8_C(  1), UINT8_C(146), UINT8_C( 40), UINT8_C(129),
                       UINT8_C( 31), UINT8_C(251), UINT8_C(213), UINT8_C( 40)),
      simde_x_vload_u8(UINT8_C(163), UINT8_C(149), UINT8_C(103), UINT8_C(189),
                       UINT8_C( 52), UINT8_C(150), UINT8_C( 23), UINT8_C(204)),
      simde_x_vload_u8(UINT8_C(114), UINT8_C(131), UINT8_C(154), UINT8_C(156),
                       UINT8_C(162), UINT8_C(214), UINT8_C(115), UINT8_C(191)) },
    { simde_x_vload_u8(UINT8_C( 23), UINT8_C(156), UINT8_C(148), UINT8_C(118),
                       UINT8_C(188), UINT8_C(  4), UINT8_C( 86), UINT8_C(160)),
      simde_x_vload_u8(UINT8_C(161), UINT8_C(134), UINT8_C(107), UINT8_C(229),
                       UINT8_C( 25), UINT8_C( 52), UINT8_C(155), UINT8_C( 87)),
      simde_x_vload_u8(UINT8_C(167), UINT8_C( 47), UINT8_C( 36), UINT8_C(246),
                       UINT8_C(113), UINT8_C( 32), UINT8_C(119), UINT8_C(196)),
      simde_x_vload_u8(UINT8_C( 29), UINT8_C( 69), UINT8_C( 77), UINT8_C(135),
                       UINT8_C( 20), UINT8_C(240), UINT8_C( 50), UINT8_C( 13)) },
    { simde_x_vload_u8(UINT8_C( 80), UINT8_C(254), UINT8_C(219), UINT8_C( 55),
                       UINT8_C(145), UINT8_C( 35), UINT8_C( 91), UINT8_C(164)),
      simde_x_vload_u8(UINT8_C(163), UINT8_C(168), UINT8_C( 74), UINT8_C(228),
                       UINT8_C(195), UINT8_C( 16), UINT8_C(221), UINT8_C( 49)),
      simde_x_vload_u8(UINT8_C(226), UINT8_C( 22), UINT8_C(115), UINT8_C(255),
                       UINT8_C(199), UINT8_C(238), UINT8_C(135), UINT8_C(189)),
      simde_x_vload_u8(UINT8_C(143), UINT8_C(108), UINT8_C(  4), UINT8_C( 82),
                       UINT8_C(149), UINT8_C(  1), UINT8_C(  5), UINT8_C( 48)) },
    { simde_x_vload_u8(UINT8_C(167), UINT8_C(125), UINT8_C(166), UINT8_C( 37),
                       UINT8_C( 32), UINT8_C( 55), UINT8_C( 47), UINT8_C( 80)),
      simde_x_vload_u8(UINT8_C(244), UINT8_C(232), UINT8_C(137), UINT8_C( 31),
                       UINT8_C( 46), UINT8_C( 91), UINT8_C( 33), UINT8_C(211)),
      simde_x_vload_u8(UINT8_C( 23), UINT8_C( 17), UINT8_C(181), UINT8_C( 53),
                       UINT8_C(140), UINT8_C( 58), UINT8_C(168), UINT8_C( 19)),
      simde_x_vload_u8(UINT8_C(202), UINT8_C(166), UINT8_C(210), UINT8_C( 59),
                       UINT8_C(126), UINT8_C( 22), UINT8_C(182), UINT8_C(144)) },
    { simde_x_vload_u8(UINT8_C(254), UINT8_C( 46), UINT8_C(143), UINT8_C(247),
                       UINT8_C(  5), UINT8_C(212), UINT8_C(221), UINT8_C(131)),
      simde_x_vload_u8(UINT8_C( 40), UINT8_C(156), UINT8_C( 80), UINT8_C(187),
                       UINT8_C( 10), UINT8_C( 30), UINT8_C( 17), UINT8_C( 96)),
      simde_x_vload_u8(UINT8_C( 10), UINT8_C(196), UINT8_C(154), UINT8_C( 88),
                       UINT8_C(102), UINT8_C(224), UINT8_C(127), UINT8_C(192)),
      simde_x_vload_u8(UINT8_C(224), UINT8_C( 86), UINT8_C(217), UINT8_C(148),
                       UINT8_C( 97), UINT8_C(150), UINT8_C( 75), UINT8_C(227)) },
    { simde_x_vload_u8(UINT8_C( 89), UINT8_C( 63), UINT8_C(164), UINT8_C(239),
                       UINT8_C(238), UINT8_C(116), UINT8_C(187), UINT8_C( 30)),
      simde_x_vload_u8(UINT8_C(222), UINT8_C(173), UINT8_C(181), UINT8_C( 57),
                       UINT8_C(142), UINT8_C(199), UINT8_C( 17), UINT8_C( 90)),
      simde_x_vload_u8(UINT8_C(134), UINT8_C(123), UINT8_C( 61), UINT8_C(108),
                       UINT8_C(114), UINT8_C(131), UINT8_C(154), UINT8_C(156)),
      simde_x_vload_u8(UINT8_C(  1), UINT8_C( 13), UINT8_C( 44), UINT8_C( 34),
                       UINT8_C(210), UINT8_C( 48), UINT8_C( 68), UINT8_C( 96)) },
    { simde_x_vload_u8(UINT8_C(213), UINT8_C( 48), UINT8_C( 81), UINT8_C(152),
                       UINT8_C(123), UINT8_C(139), UINT8_C(172), UINT8_C(180)),
      simde_x_vload_u8(UINT8_C(233), UINT8_C( 25), UINT8_C(212), UINT8_C(246),
                       UINT8_C(106), UINT8_C(222), UINT8_C( 90), UINT8_C( 32)),
      simde_x_vload_u8(UINT8_C(  2), UINT8_C(232), UINT8_C(192), UINT8_C(188),
                       UINT8_C(147), UINT8_C(192), UINT8_C(130), UINT8_C(116)),
      simde_x_vload_u8(UINT8_C(238), UINT8_C(255), UINT8_C( 61), UINT8_C( 94),
                       UINT8_C(164), UINT8_C(109), UINT8_C(212), UINT8_C(  8)) },
    { simde_x_vload_u8(UINT8_C(246), UINT8_C(243), UINT8_C(213), UINT8_C(164),
                       UINT8_C( 88), UINT8_C(171), UINT8_C(111), UINT8_C(218)),
      simde_x_vload_u8(UINT8_C(202), UINT8_C( 90), UINT8_C(241), UINT8_C(194),
                       UINT8_C(193), UINT8_C( 24), UINT8_C(214), UINT8_C(201)),
      simde_x_vload_u8(UINT8_C(194), UINT8_C(172), UINT8_C(205), UINT8_C(132),
                       UINT8_C( 44), UINT8_C(238), UINT8_C(107), UINT8_C( 72)),
      simde_x_vload_u8(UINT8_C(238), UINT8_C( 69), UINT8_C(177), UINT8_C(102),
                       UINT8_C(195), UINT8_C(129), UINT8_C(  4), UINT8_C( 89)) },
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
    { simde_x_vload_u16(UINT16_C(32976), UINT16_C(24667), UINT16_C(15245), UINT16_C( 6961)),
      simde_x_vload_u16(UINT16_C(37377), UINT16_C(33064), UINT16_C(64287), UINT16_C(10453)),
      simde_x_vload_u16(UINT16_C(38307), UINT16_C(48487), UINT16_C(38452), UINT16_C(52247)),
      simde_x_vload_u16(UINT16_C(33906), UINT16_C(40090), UINT16_C(54946), UINT16_C(48755)) },
    { simde_x_vload_u16(UINT16_C(39959), UINT16_C(30356), UINT16_C( 1212), UINT16_C(41046)),
      simde_x_vload_u16(UINT16_C(34465), UINT16_C(58731), UINT16_C(13337), UINT16_C(22427)),
      simde_x_vload_u16(UINT16_C(12199), UINT16_C(63012), UINT16_C( 8305), UINT16_C(50295)),
      simde_x_vload_u16(UINT16_C(17693), UINT16_C(34637), UINT16_C(61716), UINT16_C( 3378)) },
    { simde_x_vload_u16(UINT16_C(65104), UINT16_C(14299), UINT16_C( 9105), UINT16_C(42075)),
      simde_x_vload_u16(UINT16_C(43171), UINT16_C(58442), UINT16_C( 4291), UINT16_C(12765)),
      simde_x_vload_u16(UINT16_C( 5858), UINT16_C(65395), UINT16_C(61127), UINT16_C(48519)),
      simde_x_vload_u16(UINT16_C(27791), UINT16_C(21252), UINT16_C(  405), UINT16_C(12293)) },
    { simde_x_vload_u16(UINT16_C(32167), UINT16_C( 9638), UINT16_C(14112), UINT16_C(20527)),
      simde_x_vload_u16(UINT16_C(59636), UINT16_C( 8073), UINT16_C(23342), UINT16_C(54049)),
      simde_x_vload_u16(UINT16_C( 4375), UINT16_C(13749), UINT16_C(14988), UINT16_C( 5032)),
      simde_x_vload_u16(UINT16_C(42442), UINT16_C(15314), UINT16_C( 5758), UINT16_C(37046)) },
    { simde_x_vload_u16(UINT16_C(12030), UINT16_C(63375), UINT16_C(54277), UINT16_C(33757)),
      simde_x_vload_u16(UINT16_C(39976), UINT16_C(47952), UINT16_C( 7690), UINT16_C(24593)),
      simde_x_vload_u16(UINT16_C(50186), UINT16_C(22682), UINT16_C(57446), UINT16_C(49279)),
      simde_x_vload_u16(UINT16_C(22240), UINT16_C(38105), UINT16_C(38497), UINT16_C(58443)) },
    { simde_x_vload_u16(UINT16_C(16217), UINT16_C(61348), UINT16_C(29934), UINT16_C( 7867)),
      simde_x_vload_u16(UINT16_C(44510), UINT16_C(14773), UINT16_C(51086), UINT16_C(23057)),
      simde_x_vload_u16(UINT16_C(31622), UINT16_C(27709), UINT16_C(33650), UINT16_C(40090)),
      simde_x_vload_u16(UINT16_C( 3329), UINT16_C( 8748), UINT16_C(12498), UINT16_C(24900)) },
    { simde_x_vload_u16(UINT16_C(12501), UINT16_C(38993), UINT16_C(35707), UINT16_C(46252)),
      simde_x_vload_u16(UINT16_C( 6633), UINT16_C(63188), UINT16_C(56938), UINT16_C( 8282)),
      simde_x_vload_u16(UINT16_C(59394), UINT16_C(48320), UINT16_C(49299), UINT16_C(29826)),
      simde_x_vload_u16(UINT16_C(65262), UINT16_C(24125), UINT16_C(28068), UINT16_C( 2260)) },
    { simde_x_vload_u16(UINT16_C(62454), UINT16_C(42197), UINT16_C(43864), UINT16_C(55919)),
      simde_x_vload_u16(UINT16_C(23242), UINT16_C(49905), UINT16_C( 6337), UINT16_C(51670)),
      simde_x_vload_u16(UINT16_C(44226), UINT16_C(33997), UINT16_C(60972), UINT16_C(18539)),
      simde_x_vload_u16(UINT16_C(17902), UINT16_C(26289), UINT16_C(32963), UINT16_C(22788)) },
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
    { simde_x_vload_u32(UINT32_C(1616609488), UINT32_C( 456211341)),
      simde_x_vload_u32(UINT32_C(2166919681), UINT32_C( 685112095)),
      simde_x_vload_u32(UINT32_C(3177682339), UINT32_C(3424097844)),
      simde_x_vload_u32(UINT32_C(2627372146), UINT32_C(3195197090)) },
    { simde_x_vload_u32(UINT32_C(1989450775), UINT32_C(2689991868)),
      simde_x_vload_u32(UINT32_C(3849029281), UINT32_C(1469789209)),
      simde_x_vload_u32(UINT32_C(4129566631), UINT32_C(3296141425)),
      simde_x_vload_u32(UINT32_C(2269988125), UINT32_C( 221376788)) },
    { simde_x_vload_u32(UINT32_C( 937164368), UINT32_C(2757436305)),
      simde_x_vload_u32(UINT32_C(3830098083), UINT32_C( 836571331)),
      simde_x_vload_u32(UINT32_C(4285732578), UINT32_C(3179802311)),
      simde_x_vload_u32(UINT32_C(1392798863), UINT32_C( 805699989)) },
    { simde_x_vload_u32(UINT32_C( 631668135), UINT32_C(1345271584)),
      simde_x_vload_u32(UINT32_C( 529131764), UINT32_C(3542178606)),
      simde_x_vload_u32(UINT32_C( 901058839), UINT32_C( 329792140)),
      simde_x_vload_u32(UINT32_C(1003595210), UINT32_C(2427852414)) },
    { simde_x_vload_u32(UINT32_C(4153356030), UINT32_C(2212353029)),
      simde_x_vload_u32(UINT32_C(3142622248), UINT32_C(1611734538)),
      simde_x_vload_u32(UINT32_C(1486537738), UINT32_C(3229605990)),
      simde_x_vload_u32(UINT32_C(2497271520), UINT32_C(3830224481)) },
    { simde_x_vload_u32(UINT32_C(4020518745), UINT32_C( 515601646)),
      simde_x_vload_u32(UINT32_C( 968207838), UINT32_C(1511114638)),
      simde_x_vload_u32(UINT32_C(1815968646), UINT32_C(2627371890)),
      simde_x_vload_u32(UINT32_C( 573312257), UINT32_C(1631858898)) },
    { simde_x_vload_u32(UINT32_C(2555457749), UINT32_C(3031206779)),
      simde_x_vload_u32(UINT32_C(4141095401), UINT32_C( 542826090)),
      simde_x_vload_u32(UINT32_C(3166758914), UINT32_C(1954726035)),
      simde_x_vload_u32(UINT32_C(1581121262), UINT32_C( 148139428)) },
    { simde_x_vload_u32(UINT32_C(2765485046), UINT32_C(3664751448)),
      simde_x_vload_u32(UINT32_C(3270597322), UINT32_C(3386251457)),
      simde_x_vload_u32(UINT32_C(2228071618), UINT32_C(1215032876)),
      simde_x_vload_u32(UINT32_C(1722959342), UINT32_C(1493532867)) },
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
    { simde_x_vloadq_s8(INT8_C( -48), INT8_C(-128), INT8_C(  91), INT8_C(  96),
                        INT8_C(-115), INT8_C(  59), INT8_C(  49), INT8_C(  27),
                        INT8_C(   1), INT8_C(-110), INT8_C(  40), INT8_C(-127),
                        INT8_C(  31), INT8_C(  -5), INT8_C( -43), INT8_C(  40)),
      simde_x_vloadq_s8(INT8_C( -93), INT8_C(-107), INT8_C( 103), INT8_C( -67),
                        INT8_C(  52), INT8_C(-106), INT8_C(  23), INT8_C( -52),
                        INT8_C(  23), INT8_C(-100), INT8_C(-108), INT8_C( 118),
                        INT8_C( -68), INT8_C(   4), INT8_C(  86), INT8_C( -96)),
      simde_x_vloadq_s8(INT8_C( -95), INT8_C(-122), INT8_C( 107), INT8_C( -27),
                        INT8_C(  25), INT8_C(  52), INT8_C(-101), INT8_C(  87),
                        INT8_C( -89), INT8_C(  47), INT8_C(  36), INT8_C( -10),
                        INT8_C( 113), INT8_C(  32), INT8_C( 119), INT8_C( -60)),
      simde_x_vloadq_s8(INT8_C( -46), INT8_C(-113), INT8_C(  95), INT8_C(-120),
                        INT8_C( -88), INT8_C( -39), INT8_C( -83), INT8_C( -90),
                        INT8_C( 113), INT8_C(  37), INT8_C( -72), INT8_C(   1),
                        INT8_C( -44), INT8_C(  23), INT8_C( -10), INT8_C(  76)) },
    { simde_x_vloadq_s8(INT8_C(  80), INT8_C(  -2), INT8_C( -37), INT8_C(  55),
                        INT8_C(-111), INT8_C(  35), INT8_C(  91), INT8_C( -92),
                        INT8_C( -93), INT8_C( -88), INT8_C(  74), INT8_C( -28),
                        INT8_C( -61), INT8_C(  16), INT8_C( -35), INT8_C(  49)),
      simde_x_vloadq_s8(INT8_C( -30), INT8_C(  22), INT8_C( 115), INT8_C(  -1),
                        INT8_C( -57), INT8_C( -18), INT8_C(-121), INT8_C( -67),
                        INT8_C( -89), INT8_C( 125), INT8_C( -90), INT8_C(  37),
                        INT8_C(  32), INT8_C(  55), INT8_C(  47), INT8_C(  80)),
      simde_x_vloadq_s8(INT8_C( -12), INT8_C( -24), INT8_C(-119), INT8_C(  31),
                        INT8_C(  46), INT8_C(  91), INT8_C(  33), INT8_C( -45),
                        INT8_C(  23), INT8_C(  17), INT8_C( -75), INT8_C(  53),
                        INT8_C(-116), INT8_C(  58), INT8_C( -88), INT8_C(  19)),
      simde_x_vloadq_s8(INT8_C(  98), INT8_C(  44), INT8_C( -59), INT8_C(  87),
                        INT8_C(  -8), INT8_C(-112), INT8_C( -11), INT8_C( -70),
                        INT8_C(  19), INT8_C(  20), INT8_C(  89), INT8_C( -12),
                        INT8_C(  87), INT8_C(  19), INT8_C( 100), INT8_C( 110)) },
    { simde_x_vloadq_s8(INT8_C(  -2), INT8_C(  46), INT8_C(-113), INT8_C(  -9),
                        INT8_C(   5), INT8_C( -44), INT8_C( -35), INT8_C(-125),
                        INT8_C(  40), INT8_C(-100), INT8_C(  80), INT8_C( -69),
                        INT8_C(  10), INT8_C(  30), INT8_C(  17), INT8_C(  96)),
      simde_x_vloadq_s8(INT8_C(  10), INT8_C( -60), INT8_C(-102), INT8_C(  88),
                        INT8_C( 102), INT8_C( -32), INT8_C( 127), INT8_C( -64),
                        INT8_C(  89), INT8_C(  63), INT8_C( -92), INT8_C( -17),
                        INT8_C( -18), INT8_C( 116), INT8_C( -69), INT8_C(  30)),
      simde_x_vloadq_s8(INT8_C( -34), INT8_C( -83), INT8_C( -75), INT8_C(  57),
                        INT8_C(-114), INT8_C( -57), INT8_C(  17), INT8_C(  90),
                        INT8_C(-122), INT8_C( 123), INT8_C(  61), INT8_C( 108),
                        INT8_C( 114), INT8_C(-125), INT8_C(-102), INT8_C(-100)),
      simde_x_vloadq_s8(INT8_C(  42), INT8_C(  69), INT8_C( -86), INT8_C(  22),
                        INT8_C( -35), INT8_C( -19), INT8_C(  75), INT8_C(  29),
                        INT8_C(  -5), INT8_C( -40), INT8_C( -23), INT8_C(  56),
                        INT8_C(-114), INT8_C(  15), INT8_C(  50), INT8_C( -30)) },
    { simde_x_vloadq_s8(INT8_C( -43), INT8_C(  48), INT8_C(  81), INT8_C(-104),
                        INT8_C( 123), INT8_C(-117), INT8_C( -84), INT8_C( -76),
                        INT8_C( -23), INT8_C(  25), INT8_C( -44), INT8_C( -10),
                        INT8_C( 106), INT8_C( -34), INT8_C(  90), INT8_C(  32)),
      simde_x_vloadq_s8(INT8_C(   2), INT8_C( -24), INT8_C( -64), INT8_C( -68),
                        INT8_C(-109), INT8_C( -64), INT8_C(-126), INT8_C( 116),
                        INT8_C( -10), INT8_C( -13), INT8_C( -43), INT8_C( -92),
                        INT8_C(  88), INT8_C( -85), INT8_C( 111), INT8_C( -38)),
      simde_x_vloadq_s8(INT8_C( -54), INT8_C(  90), INT8_C( -15), INT8_C( -62),
                        INT8_C( -63), INT8_C(  24), INT8_C( -42), INT8_C( -55),
                        INT8_C( -62), INT8_C( -84), INT8_C( -51), INT8_C(-124),
                        INT8_C(  44), INT8_C( -18), INT8_C( 107), INT8_C(  72)),
      simde_x_vloadq_s8(INT8_C(  13), INT8_C( -94), INT8_C(-126), INT8_C( -98),
                        INT8_C( -87), INT8_C( -29), INT8_C(   0), INT8_C(  95),
                        INT8_C(  29), INT8_C(  96), INT8_C( -36), INT8_C(  22),
                        INT8_C(-106), INT8_C(  33), INT8_C(  94), INT8_C(-114)) },
    { simde_x_vloadq_s8(INT8_C( 127), INT8_C(  49), INT8_C( -20), INT8_C(  97),
                        INT8_C( -30), INT8_C(  -8), INT8_C(  99), INT8_C( -64),
                        INT8_C(-101), INT8_C( -79), INT8_C(  -7), INT8_C( 122),
                        INT8_C( -65), INT8_C( 106), INT8_C(  41), INT8_C(  -1)),
      simde_x_vloadq_s8(INT8_C( -17), INT8_C(  -1), INT8_C( 108), INT8_C(  81),
                        INT8_C(  63), INT8_C(  -2), INT8_C(  57), INT8_C(   4),
                        INT8_C(  85), INT8_C(  -8), INT8_C( -24), INT8_C(  50),
                        INT8_C(-101), INT8_C(  -9), INT8_C(  50), INT8_C( -11)),
      simde_x_vloadq_s8(INT8_C( -42), INT8_C( 126), INT8_C(  88), INT8_C( -85),
                        INT8_C( -96), INT8_C( 105), INT8_C( -75), INT8_C(  73),
                        INT8_C(  10), INT8_C(  94), INT8_C(-115), INT8_C(-115),
                        INT8_C(-121), INT8_C( -19), INT8_C( -10), INT8_C( 116)),
      simde_x_vloadq_s8(INT8_C(-104), INT8_C( -80), INT8_C(   0), INT8_C(   7),
                        INT8_C(-127), INT8_C(  99), INT8_C( -25), INT8_C(   5),
                        INT8_C( -26), INT8_C(  23), INT8_C(  84), INT8_C(  31),
                        INT8_C( -45), INT8_C( 116), INT8_C( 101), INT8_C( 126)) },
    { simde_x_vloadq_s8(INT8_C( 105), INT8_C( -47), INT8_C( 108), INT8_C( 118),
                        INT8_C( 119), INT8_C(  30), INT8_C(  89), INT8_C(  -6),
                        INT8_C(  74), INT8_C( 119), INT8_C( -73), INT8_C( 110),
                        INT8_C( 122), INT8_C(  95), INT8_C( 105), INT8_C(  39)),
      simde_x_vloadq_s8(INT8_C(   0), INT8_C( -15), INT8_C(  82), INT8_C(-113),
                        INT8_C(-123), INT8_C(-106), INT8_C( -52), INT8_C( -12),
                        INT8_C(  54), INT8_C(-123), INT8_C( 109), INT8_C( -24),
                        INT8_C( 123), INT8_C(  65), INT8_C(  59), INT8_C( -80)),
      simde_x_vloadq_s8(INT8_C( -46), INT8_C( -98), INT8_C( -56), INT8_C(  56),
                        INT8_C(-109), INT8_C(  69), INT8_C(  10), INT8_C( -13),
                        INT8_C( -58), INT8_C(  95), INT8_C(  54), INT8_C( -52),
                        INT8_C(-119), INT8_C( -21), INT8_C( 116), INT8_C(-121)),
      simde_x_vloadq_s8(INT8_C(-105), INT8_C(  36), INT8_C( -10), INT8_C(  31),
                        INT8_C(-123), INT8_C( -51), INT8_C(-105), INT8_C(  -5),
                        INT8_C( -70), INT8_C(  81), INT8_C( -18), INT8_C(-118),
                        INT8_C( 108), INT8_C( -75), INT8_C( -94), INT8_C(  80)) },
    { simde_x_vloadq_s8(INT8_C(  33), INT8_C( -77), INT8_C( 108), INT8_C( 100),
                        INT8_C( -66), INT8_C( -94), INT8_C( -85), INT8_C(  91),
                        INT8_C( -44), INT8_C(  79), INT8_C(  33), INT8_C(  32),
                        INT8_C(   7), INT8_C(  22), INT8_C(  94), INT8_C( -34)),
      simde_x_vloadq_s8(INT8_C( -98), INT8_C(  14), INT8_C( 101), INT8_C(-120),
                        INT8_C( 103), INT8_C(  12), INT8_C( -80), INT8_C(  11),
                        INT8_C(  39), INT8_C( -39), INT8_C(  92), INT8_C( -47),
                        INT8_C(  21), INT8_C( -57), INT8_C(   1), INT8_C(  55)),
      simde_x_vloadq_s8(INT8_C(  56), INT8_C(   1), INT8_C(-108), INT8_C(  73),
                        INT8_C(-115), INT8_C(  27), INT8_C( -49), INT8_C(  77),
                        INT8_C( -54), INT8_C( 107), INT8_C( -79), INT8_C(  22),
                        INT8_C(  41), INT8_C(  84), INT8_C( 115), INT8_C(  45)),
      simde_x_vloadq_s8(INT8_C( -69), INT8_C( -64), INT8_C(  61), INT8_C(  37),
                        INT8_C(-104), INT8_C( -79), INT8_C( -54), INT8_C( -99),
                        INT8_C(  49), INT8_C( -31), INT8_C( -52), INT8_C( 101),
                        INT8_C(  27), INT8_C( -93), INT8_C( -48), INT8_C( -24)) },
    { simde_x_vloadq_s8(INT8_C(-113), INT8_C( -55), INT8_C( -45), INT8_C( -90),
                        INT8_C(   4), INT8_C( -91), INT8_C(-104), INT8_C(  17),
                        INT8_C(  24), INT8_C(  41), INT8_C( -13), INT8_C(  90),
                        INT8_C( 118), INT8_C( -20), INT8_C(  16), INT8_C( 118)),
      simde_x_vloadq_s8(INT8_C( 125), INT8_C(  61), INT8_C( 124), INT8_C( -60),
                        INT8_C(  -5), INT8_C( -36), INT8_C( -41), INT8_C( -33),
                        INT8_C(  15), INT8_C(  -9), INT8_C(  85), INT8_C(   4),
                        INT8_C(   0), INT8_C( -94), INT8_C(-105), INT8_C(  61)),
      simde_x_vloadq_s8(INT8_C(  -9), INT8_C(  -7), INT8_C( -56), INT8_C( -42),
                        INT8_C( -17), INT8_C( -26), INT8_C(  10), INT8_C(  35),
                        INT8_C(  50), INT8_C( -35), INT8_C(  16), INT8_C(  55),
                        INT8_C(  65), INT8_C(  75), INT8_C(-103), INT8_C(  84)),
      simde_x_vloadq_s8(INT8_C(  21), INT8_C(  13), INT8_C(-121), INT8_C( -72),
                        INT8_C(  16), INT8_C( -81), INT8_C( -53), INT8_C(  85),
                        INT8_C(  59), INT8_C(  67), INT8_C(  56), INT8_C(-115),
                        INT8_C( -73), INT8_C(-107), INT8_C(  18), INT8_C(-115)) },
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
    { simde_x_vloadq_s16(INT16_C(-32560), INT16_C( 24667), INT16_C( 15245), INT16_C(  6961),
                         INT16_C(-28159), INT16_C(-32472), INT16_C( -1249), INT16_C( 10453)),
      simde_x_vloadq_s16(INT16_C(-27229), INT16_C(-17049), INT16_C(-27084), INT16_C(-13289),
                         INT16_C(-25577), INT16_C( 30356), INT16_C(  1212), INT16_C(-24490)),
      simde_x_vloadq_s16(INT16_C(-31071), INT16_C( -6805), INT16_C( 13337), INT16_C( 22427),
                         INT16_C( 12199), INT16_C( -2524), INT16_C(  8305), INT16_C(-15241)),
      simde_x_vloadq_s16(INT16_C(-28718), INT16_C(-30625), INT16_C( -9870), INT16_C(-22859),
                         INT16_C(  9617), INT16_C(   408), INT16_C(  5844), INT16_C( 19702)) },
    { simde_x_vloadq_s16(INT16_C(  -432), INT16_C( 14299), INT16_C(  9105), INT16_C(-23461),
                         INT16_C(-22365), INT16_C( -7094), INT16_C(  4291), INT16_C( 12765)),
      simde_x_vloadq_s16(INT16_C(  5858), INT16_C(  -141), INT16_C( -4409), INT16_C(-17017),
                         INT16_C( 32167), INT16_C(  9638), INT16_C( 14112), INT16_C( 20527)),
      simde_x_vloadq_s16(INT16_C( -5900), INT16_C(  8073), INT16_C( 23342), INT16_C(-11487),
                         INT16_C(  4375), INT16_C( 13749), INT16_C( 14988), INT16_C(  5032)),
      simde_x_vloadq_s16(INT16_C( 11326), INT16_C( 22513), INT16_C(-28680), INT16_C(-17931),
                         INT16_C(  5427), INT16_C( -2983), INT16_C(  5167), INT16_C( 28260)) },
    { simde_x_vloadq_s16(INT16_C( 12030), INT16_C( -2161), INT16_C(-11259), INT16_C(-31779),
                         INT16_C(-25560), INT16_C(-17584), INT16_C(  7690), INT16_C( 24593)),
      simde_x_vloadq_s16(INT16_C(-15350), INT16_C( 22682), INT16_C( -8090), INT16_C(-16257),
                         INT16_C( 16217), INT16_C( -4188), INT16_C( 29934), INT16_C(  7867)),
      simde_x_vloadq_s16(INT16_C(-21026), INT16_C( 14773), INT16_C(-14450), INT16_C( 23057),
                         INT16_C( 31622), INT16_C( 27709), INT16_C(-31886), INT16_C(-25446)),
      simde_x_vloadq_s16(INT16_C( 17706), INT16_C(  5748), INT16_C( -4899), INT16_C(  7535),
                         INT16_C(-10155), INT16_C( 14313), INT16_C(  3974), INT16_C( -7630)) },
    { simde_x_vloadq_s16(INT16_C( 12501), INT16_C(-26543), INT16_C(-29829), INT16_C(-19284),
                         INT16_C(  6633), INT16_C( -2348), INT16_C( -8598), INT16_C(  8282)),
      simde_x_vloadq_s16(INT16_C( -6142), INT16_C(-17216), INT16_C(-16237), INT16_C( 29826),
                         INT16_C( -3082), INT16_C(-23339), INT16_C(-21672), INT16_C( -9617)),
      simde_x_vloadq_s16(INT16_C( 23242), INT16_C(-15631), INT16_C(  6337), INT16_C(-13866),
                         INT16_C(-21310), INT16_C(-31539), INT16_C( -4564), INT16_C( 18539)),
      simde_x_vloadq_s16(INT16_C(-23651), INT16_C(-24958), INT16_C( -7255), INT16_C( 24408),
                         INT16_C( 24861), INT16_C(  5852), INT16_C(  8510), INT16_C(-29098)) },
    { simde_x_vloadq_s16(INT16_C( 12671), INT16_C( 25068), INT16_C( -1822), INT16_C(-16285),
                         INT16_C(-20069), INT16_C( 31481), INT16_C( 27327), INT16_C(  -215)),
      simde_x_vloadq_s16(INT16_C(   -17), INT16_C( 20844), INT16_C(  -449), INT16_C(  1081),
                         INT16_C( -1963), INT16_C( 13032), INT16_C( -2149), INT16_C( -2766)),
      simde_x_vloadq_s16(INT16_C( 32470), INT16_C(-21672), INT16_C( 27040), INT16_C( 18869),
                         INT16_C( 24074), INT16_C(-29299), INT16_C( -4729), INT16_C( 29942)),
      simde_x_vloadq_s16(INT16_C(-20378), INT16_C(  2048), INT16_C( 25667), INT16_C(  1503),
                         INT16_C(  5968), INT16_C(  8276), INT16_C( 29907), INT16_C( 32493)) },
    { simde_x_vloadq_s16(INT16_C(-11927), INT16_C( 30316), INT16_C(  7799), INT16_C( -1447),
                         INT16_C( 30538), INT16_C( 28343), INT16_C( 24442), INT16_C( 10089)),
      simde_x_vloadq_s16(INT16_C( -3840), INT16_C(-28846), INT16_C(-27003), INT16_C( -2868),
                         INT16_C(-31434), INT16_C( -6035), INT16_C( 16763), INT16_C(-20421)),
      simde_x_vloadq_s16(INT16_C(-24878), INT16_C( 14536), INT16_C( 17811), INT16_C( -3318),
                         INT16_C( 24518), INT16_C(-13258), INT16_C( -5239), INT16_C(-30860)),
      simde_x_vloadq_s16(INT16_C(  9111), INT16_C(  8162), INT16_C(-12923), INT16_C(  -997),
                         INT16_C( 20954), INT16_C(-29970), INT16_C(-19092), INT16_C( 20528)) },
    { simde_x_vloadq_s16(INT16_C(-19679), INT16_C( 25708), INT16_C(-23874), INT16_C( 23467),
                         INT16_C( 20436), INT16_C(  8225), INT16_C(  5639), INT16_C( -8610)),
      simde_x_vloadq_s16(INT16_C(  3742), INT16_C(-30619), INT16_C(  3175), INT16_C(  2992),
                         INT16_C( -9945), INT16_C(-11940), INT16_C(-14571), INT16_C( 14081)),
      simde_x_vloadq_s16(INT16_C(   312), INT16_C( 18836), INT16_C(  7053), INT16_C( 19919),
                         INT16_C( 27594), INT16_C(  5809), INT16_C( 21545), INT16_C( 11635)),
      simde_x_vloadq_s16(INT16_C(-16249), INT16_C(  9627), INT16_C(-19996), INT16_C(-25142),
                         INT16_C( -7561), INT16_C( 25974), INT16_C(-23781), INT16_C( -6164)) },
    { simde_x_vloadq_s16(INT16_C(-13937), INT16_C(-22829), INT16_C(-23292), INT16_C(  4504),
                         INT16_C( 10520), INT16_C( 23283), INT16_C( -5002), INT16_C( 30224)),
      simde_x_vloadq_s16(INT16_C( 15741), INT16_C(-15236), INT16_C( -8965), INT16_C( -8233),
                         INT16_C( -2289), INT16_C(  1109), INT16_C(-24064), INT16_C( 15767)),
      simde_x_vloadq_s16(INT16_C( -1545), INT16_C(-10552), INT16_C( -6417), INT16_C(  8970),
                         INT16_C( -8910), INT16_C( 14096), INT16_C( 19265), INT16_C( 21657)),
      simde_x_vloadq_s16(INT16_C(  3349), INT16_C(-18145), INT16_C(-20744), INT16_C( 21707),
                         INT16_C( 17141), INT16_C(-29266), INT16_C(-27209), INT16_C(-29422)) },
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
    { simde_x_vloadq_s32(INT32_C( 1616609488), INT32_C(  456211341),
                         INT32_C(-2128047615), INT32_C(  685112095)),
      simde_x_vloadq_s32(INT32_C(-1117284957), INT32_C( -870869452),
                         INT32_C( 1989450775), INT32_C(-1604975428)),
      simde_x_vloadq_s32(INT32_C( -445938015), INT32_C( 1469789209),
                         INT32_C( -165400665), INT32_C( -998825871)),
      simde_x_vloadq_s32(INT32_C(-2007010866), INT32_C(-1884447320),
                         INT32_C(   12068241), INT32_C( 1291261652)) },
    { simde_x_vloadq_s32(INT32_C(  937164368), INT32_C(-1537530991),
                         INT32_C( -464869213), INT32_C(  836571331)),
      simde_x_vloadq_s32(INT32_C(   -9234718), INT32_C(-1115164985),
                         INT32_C(  631668135), INT32_C( 1345271584)),
      simde_x_vloadq_s32(INT32_C(  529131764), INT32_C( -752788690),
                         INT32_C(  901058839), INT32_C(  329792140)),
      simde_x_vloadq_s32(INT32_C( 1475530850), INT32_C(-1175154696),
                         INT32_C( -195478509), INT32_C( 1852050775)) },
    { simde_x_vloadq_s32(INT32_C( -141611266), INT32_C(-2082614267),
                         INT32_C(-1152345048), INT32_C( 1611734538)),
      simde_x_vloadq_s32(INT32_C( 1486537738), INT32_C(-1065361306),
                         INT32_C( -274448551), INT32_C(  515601646)),
      simde_x_vloadq_s32(INT32_C(  968207838), INT32_C( 1511114638),
                         INT32_C( 1815968646), INT32_C(-1667595406)),
      simde_x_vloadq_s32(INT32_C(  376718634), INT32_C( -364122915),
                         INT32_C(  938072149), INT32_C( -571462514)) },
    { simde_x_vloadq_s32(INT32_C(-1739509547), INT32_C(-1263760517),
                         INT32_C( -153871895), INT32_C(  542826090)),
      simde_x_vloadq_s32(INT32_C(-1128208382), INT32_C( 1954726035),
                         INT32_C(-1529482250), INT32_C( -630215848)),
      simde_x_vloadq_s32(INT32_C(-1024369974), INT32_C( -908715839),
                         INT32_C(-2066895678), INT32_C( 1215032876)),
      simde_x_vloadq_s32(INT32_C(-1635671139), INT32_C(  167764905),
                         INT32_C(  383541533), INT32_C(-1906892482)) },
    { simde_x_vloadq_s32(INT32_C( 1642869119), INT32_C(-1067190046),
                         INT32_C( 2063184283), INT32_C(  -14062913)),
      simde_x_vloadq_s32(INT32_C( 1366097903), INT32_C(   70909503),
                         INT32_C(  854128725), INT32_C( -181209189)),
      simde_x_vloadq_s32(INT32_C(-1420263722), INT32_C( 1236625824),
                         INT32_C(-1920115190), INT32_C( 1962339719)),
      simde_x_vloadq_s32(INT32_C(-1143492506), INT32_C(   98526275),
                         INT32_C( -711059632), INT32_C( 2129485995)) },
    { simde_x_vloadq_s32(INT32_C( 1986842985), INT32_C(  -94822793),
                         INT32_C( 1857517386), INT32_C(  661217146)),
      simde_x_vloadq_s32(INT32_C(-1890389760), INT32_C( -187918715),
                         INT32_C( -395475658), INT32_C(-1338293893)),
      simde_x_vloadq_s32(INT32_C(  952671954), INT32_C( -217430637),
                         INT32_C( -868851770), INT32_C(-2022380663)),
      simde_x_vloadq_s32(INT32_C( -856218729), INT32_C(  -65310871),
                         INT32_C(-1964073798), INT32_C( 1345303916)) },
    { simde_x_vloadq_s32(INT32_C( 1684845345), INT32_C( 1537974974),
                         INT32_C(  539054036), INT32_C( -564259321)),
      simde_x_vloadq_s32(INT32_C(-2006643042), INT32_C(  196086887),
                         INT32_C( -782444249), INT32_C(  922863381)),
      simde_x_vloadq_s32(INT32_C( 1234436408), INT32_C( 1305418637),
                         INT32_C(  380726218), INT32_C(  762532905)),
      simde_x_vloadq_s32(INT32_C(-1556234105), INT32_C(-1647660572),
                         INT32_C( 1702224503), INT32_C( -403928845)) },
    { simde_x_vloadq_s32(INT32_C(-1496069745), INT32_C(  295216388),
                         INT32_C( 1525885208), INT32_C( 1980820598)),
      simde_x_vloadq_s32(INT32_C( -998490755), INT32_C( -539501317),
                         INT32_C(   72742671), INT32_C( 1033347584)),
      simde_x_vloadq_s32(INT32_C( -691471881), INT32_C(  587917039),
                         INT32_C(  923852082), INT32_C( 1419332417)),
      simde_x_vloadq_s32(INT32_C(-1189050871), INT32_C( 1422634744),
                         INT32_C(-1917972677), INT32_C(-1928161865)) },
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
    { simde_x_vloadq_u8(UINT8_C(208), UINT8_C(128), UINT8_C( 91), UINT8_C( 96),
                        UINT8_C(141), UINT8_C( 59), UINT8_C( 49), UINT8_C( 27),
                        UINT8_C(  1), UINT8_C(146), UINT8_C( 40), UINT8_C(129),
                        UINT8_C( 31), UINT8_C(251), UINT8_C(213), UINT8_C( 40) ),
      simde_x_vloadq_u8(UINT8_C(163), UINT8_C(149), UINT8_C(103), UINT8_C(189),
                        UINT8_C( 52), UINT8_C(150), UINT8_C( 23), UINT8_C(204),
                        UINT8_C( 23), UINT8_C(156), UINT8_C(148), UINT8_C(118),
                        UINT8_C(188), UINT8_C(  4), UINT8_C( 86), UINT8_C(160) ),
      simde_x_vloadq_u8(UINT8_C(161), UINT8_C(134), UINT8_C(107), UINT8_C(229),
                        UINT8_C( 25), UINT8_C( 52), UINT8_C(155), UINT8_C( 87),
                        UINT8_C(167), UINT8_C( 47), UINT8_C( 36), UINT8_C(246),
                        UINT8_C(113), UINT8_C( 32), UINT8_C(119), UINT8_C(196) ),
      simde_x_vloadq_u8(UINT8_C(206), UINT8_C(113), UINT8_C( 95), UINT8_C(136),
                        UINT8_C(114), UINT8_C(217), UINT8_C(181), UINT8_C(166),
                        UINT8_C(145), UINT8_C( 37), UINT8_C(184), UINT8_C(  1),
                        UINT8_C(212), UINT8_C( 23), UINT8_C(246), UINT8_C( 76) ) },
    { simde_x_vloadq_u8(UINT8_C( 80), UINT8_C(254), UINT8_C(219), UINT8_C( 55),
                        UINT8_C(145), UINT8_C( 35), UINT8_C( 91), UINT8_C(164),
                        UINT8_C(163), UINT8_C(168), UINT8_C( 74), UINT8_C(228),
                        UINT8_C(195), UINT8_C( 16), UINT8_C(221), UINT8_C( 49) ),
      simde_x_vloadq_u8(UINT8_C(226), UINT8_C( 22), UINT8_C(115), UINT8_C(255),
                        UINT8_C(199), UINT8_C(238), UINT8_C(135), UINT8_C(189),
                        UINT8_C(167), UINT8_C(125), UINT8_C(166), UINT8_C( 37),
                        UINT8_C( 32), UINT8_C( 55), UINT8_C( 47), UINT8_C( 80) ),
      simde_x_vloadq_u8(UINT8_C(244), UINT8_C(232), UINT8_C(137), UINT8_C( 31),
                        UINT8_C( 46), UINT8_C( 91), UINT8_C( 33), UINT8_C(211),
                        UINT8_C( 23), UINT8_C( 17), UINT8_C(181), UINT8_C( 53),
                        UINT8_C(140), UINT8_C( 58), UINT8_C(168), UINT8_C( 19) ),
      simde_x_vloadq_u8(UINT8_C( 98), UINT8_C(208), UINT8_C(241), UINT8_C( 87),
                        UINT8_C(248), UINT8_C(144), UINT8_C(245), UINT8_C(186),
                        UINT8_C( 19), UINT8_C( 60), UINT8_C( 89), UINT8_C(244),
                        UINT8_C( 47), UINT8_C( 19), UINT8_C( 86), UINT8_C(244) ) },
    { simde_x_vloadq_u8(UINT8_C(254), UINT8_C( 46), UINT8_C(143), UINT8_C(247),
                        UINT8_C(  5), UINT8_C(212), UINT8_C(221), UINT8_C(131),
                        UINT8_C( 40), UINT8_C(156), UINT8_C( 80), UINT8_C(187),
                        UINT8_C( 10), UINT8_C( 30), UINT8_C( 17), UINT8_C( 96) ),
      simde_x_vloadq_u8(UINT8_C( 10), UINT8_C(196), UINT8_C(154), UINT8_C( 88),
                        UINT8_C(102), UINT8_C(224), UINT8_C(127), UINT8_C(192),
                        UINT8_C( 89), UINT8_C( 63), UINT8_C(164), UINT8_C(239),
                        UINT8_C(238), UINT8_C(116), UINT8_C(187), UINT8_C( 30) ),
      simde_x_vloadq_u8(UINT8_C(222), UINT8_C(173), UINT8_C(181), UINT8_C( 57),
                        UINT8_C(142), UINT8_C(199), UINT8_C( 17), UINT8_C( 90),
                        UINT8_C(134), UINT8_C(123), UINT8_C( 61), UINT8_C(108),
                        UINT8_C(114), UINT8_C(131), UINT8_C(154), UINT8_C(156) ),
      simde_x_vloadq_u8(UINT8_C(210), UINT8_C( 23), UINT8_C(170), UINT8_C(216),
                        UINT8_C( 45), UINT8_C(187), UINT8_C(111), UINT8_C( 29),
                        UINT8_C( 85), UINT8_C(216), UINT8_C(233), UINT8_C( 56),
                        UINT8_C(142), UINT8_C( 45), UINT8_C(240), UINT8_C(222) ) },
    { simde_x_vloadq_u8(UINT8_C(213), UINT8_C( 48), UINT8_C( 81), UINT8_C(152),
                        UINT8_C(123), UINT8_C(139), UINT8_C(172), UINT8_C(180),
                        UINT8_C(233), UINT8_C( 25), UINT8_C(212), UINT8_C(246),
                        UINT8_C(106), UINT8_C(222), UINT8_C( 90), UINT8_C( 32) ),
      simde_x_vloadq_u8(UINT8_C(  2), UINT8_C(232), UINT8_C(192), UINT8_C(188),
                        UINT8_C(147), UINT8_C(192), UINT8_C(130), UINT8_C(116),
                        UINT8_C(246), UINT8_C(243), UINT8_C(213), UINT8_C(164),
                        UINT8_C( 88), UINT8_C(171), UINT8_C(111), UINT8_C(218) ),
      simde_x_vloadq_u8(UINT8_C(202), UINT8_C( 90), UINT8_C(241), UINT8_C(194),
                        UINT8_C(193), UINT8_C( 24), UINT8_C(214), UINT8_C(201),
                        UINT8_C(194), UINT8_C(172), UINT8_C(205), UINT8_C(132),
                        UINT8_C( 44), UINT8_C(238), UINT8_C(107), UINT8_C( 72) ),
      simde_x_vloadq_u8(UINT8_C(157), UINT8_C(162), UINT8_C(130), UINT8_C(158),
                        UINT8_C(169), UINT8_C(227), UINT8_C(  0), UINT8_C(  9),
                        UINT8_C(181), UINT8_C(210), UINT8_C(204), UINT8_C(214),
                        UINT8_C( 62), UINT8_C( 33), UINT8_C( 86), UINT8_C(142) ) },
    { simde_x_vloadq_u8(UINT8_C(127), UINT8_C( 49), UINT8_C(236), UINT8_C( 97),
                        UINT8_C(226), UINT8_C(248), UINT8_C( 99), UINT8_C(192),
                        UINT8_C(155), UINT8_C(177), UINT8_C(249), UINT8_C(122),
                        UINT8_C(191), UINT8_C(106), UINT8_C( 41), UINT8_C(255) ),
      simde_x_vloadq_u8(UINT8_C(239), UINT8_C(255), UINT8_C(108), UINT8_C( 81),
                        UINT8_C( 63), UINT8_C(254), UINT8_C( 57), UINT8_C(  4),
                        UINT8_C( 85), UINT8_C(248), UINT8_C(232), UINT8_C( 50),
                        UINT8_C(155), UINT8_C(247), UINT8_C( 50), UINT8_C(245) ),
      simde_x_vloadq_u8(UINT8_C(214), UINT8_C(126), UINT8_C( 88), UINT8_C(171),
                        UINT8_C(160), UINT8_C(105), UINT8_C(181), UINT8_C( 73),
                        UINT8_C( 10), UINT8_C( 94), UINT8_C(141), UINT8_C(141),
                        UINT8_C(135), UINT8_C(237), UINT8_C(246), UINT8_C(116) ),
      simde_x_vloadq_u8(UINT8_C(102), UINT8_C(176), UINT8_C(216), UINT8_C(187),
                        UINT8_C( 67), UINT8_C( 99), UINT8_C(223), UINT8_C(  5),
                        UINT8_C( 80), UINT8_C( 23), UINT8_C(158), UINT8_C(213),
                        UINT8_C(171), UINT8_C( 96), UINT8_C(237), UINT8_C(126) ) },
    { simde_x_vloadq_u8(UINT8_C(105), UINT8_C(209), UINT8_C(108), UINT8_C(118),
                        UINT8_C(119), UINT8_C( 30), UINT8_C( 89), UINT8_C(250),
                        UINT8_C( 74), UINT8_C(119), UINT8_C(183), UINT8_C(110),
                        UINT8_C(122), UINT8_C( 95), UINT8_C(105), UINT8_C( 39) ),
      simde_x_vloadq_u8(UINT8_C(  0), UINT8_C(241), UINT8_C( 82), UINT8_C(143),
                        UINT8_C(133), UINT8_C(150), UINT8_C(204), UINT8_C(244),
                        UINT8_C( 54), UINT8_C(133), UINT8_C(109), UINT8_C(232),
                        UINT8_C(123), UINT8_C( 65), UINT8_C( 59), UINT8_C(176) ),
      simde_x_vloadq_u8(UINT8_C(210), UINT8_C(158), UINT8_C(200), UINT8_C( 56),
                        UINT8_C(147), UINT8_C( 69), UINT8_C( 10), UINT8_C(243),
                        UINT8_C(198), UINT8_C( 95), UINT8_C( 54), UINT8_C(204),
                        UINT8_C(137), UINT8_C(235), UINT8_C(116), UINT8_C(135) ),
      simde_x_vloadq_u8(UINT8_C( 59), UINT8_C(126), UINT8_C(226), UINT8_C( 31),
                        UINT8_C(133), UINT8_C(205), UINT8_C(151), UINT8_C(249),
                        UINT8_C(218), UINT8_C( 81), UINT8_C(128), UINT8_C( 82),
                        UINT8_C(136), UINT8_C(  9), UINT8_C(162), UINT8_C(254) ) },
    { simde_x_vloadq_u8(UINT8_C( 33), UINT8_C(179), UINT8_C(108), UINT8_C(100),
                        UINT8_C(190), UINT8_C(162), UINT8_C(171), UINT8_C( 91),
                        UINT8_C(212), UINT8_C( 79), UINT8_C( 33), UINT8_C( 32),
                        UINT8_C(  7), UINT8_C( 22), UINT8_C( 94), UINT8_C(222) ),
      simde_x_vloadq_u8(UINT8_C(158), UINT8_C( 14), UINT8_C(101), UINT8_C(136),
                        UINT8_C(103), UINT8_C( 12), UINT8_C(176), UINT8_C( 11),
                        UINT8_C( 39), UINT8_C(217), UINT8_C( 92), UINT8_C(209),
                        UINT8_C( 21), UINT8_C(199), UINT8_C(  1), UINT8_C( 55) ),
      simde_x_vloadq_u8(UINT8_C( 56), UINT8_C(  1), UINT8_C(148), UINT8_C( 73),
                        UINT8_C(141), UINT8_C( 27), UINT8_C(207), UINT8_C( 77),
                        UINT8_C(202), UINT8_C(107), UINT8_C(177), UINT8_C( 22),
                        UINT8_C( 41), UINT8_C( 84), UINT8_C(115), UINT8_C( 45) ),
      simde_x_vloadq_u8(UINT8_C(187), UINT8_C(166), UINT8_C(155), UINT8_C( 37),
                        UINT8_C(228), UINT8_C(177), UINT8_C(202), UINT8_C(157),
                        UINT8_C(119), UINT8_C(225), UINT8_C(118), UINT8_C(101),
                        UINT8_C( 27), UINT8_C(163), UINT8_C(208), UINT8_C(212) ) },
    { simde_x_vloadq_u8(UINT8_C(143), UINT8_C(201), UINT8_C(211), UINT8_C(166),
                        UINT8_C(  4), UINT8_C(165), UINT8_C(152), UINT8_C( 17),
                        UINT8_C( 24), UINT8_C( 41), UINT8_C(243), UINT8_C( 90),
                        UINT8_C(118), UINT8_C(236), UINT8_C( 16), UINT8_C(118) ),
      simde_x_vloadq_u8(UINT8_C(125), UINT8_C( 61), UINT8_C(124), UINT8_C(196),
                        UINT8_C(251), UINT8_C(220), UINT8_C(215), UINT8_C(223),
                        UINT8_C( 15), UINT8_C(247), UINT8_C( 85), UINT8_C(  4),
                        UINT8_C(  0), UINT8_C(162), UINT8_C(151), UINT8_C( 61) ),
      simde_x_vloadq_u8(UINT8_C(247), UINT8_C(249), UINT8_C(200), UINT8_C(214),
                        UINT8_C(239), UINT8_C(230), UINT8_C( 10), UINT8_C( 35),
                        UINT8_C( 50), UINT8_C(221), UINT8_C( 16), UINT8_C( 55),
                        UINT8_C( 65), UINT8_C( 75), UINT8_C(153), UINT8_C( 84) ),
      simde_x_vloadq_u8(UINT8_C(  9), UINT8_C(133), UINT8_C( 31), UINT8_C(184),
                        UINT8_C(248), UINT8_C(175), UINT8_C(203), UINT8_C( 85),
                        UINT8_C( 59), UINT8_C( 15), UINT8_C(174), UINT8_C(141),
                        UINT8_C(183), UINT8_C(149), UINT8_C( 18), UINT8_C(141) ) },
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
    { simde_x_vloadq_u16(UINT16_C(32976), UINT16_C(24667), UINT16_C(15245), UINT16_C( 6961),
                         UINT16_C(37377), UINT16_C(33064), UINT16_C(64287), UINT16_C(10453)),
      simde_x_vloadq_u16(UINT16_C(38307), UINT16_C(48487), UINT16_C(38452), UINT16_C(52247),
                         UINT16_C(39959), UINT16_C(30356), UINT16_C( 1212), UINT16_C(41046)),
      simde_x_vloadq_u16(UINT16_C(34465), UINT16_C(58731), UINT16_C(13337), UINT16_C(22427),
                         UINT16_C(12199), UINT16_C(63012), UINT16_C( 8305), UINT16_C(50295)),
      simde_x_vloadq_u16(UINT16_C(29134), UINT16_C(34911), UINT16_C(55666), UINT16_C(42677),
                         UINT16_C( 9617), UINT16_C(  184), UINT16_C( 5844), UINT16_C(19702)) },
    { simde_x_vloadq_u16(UINT16_C(65104), UINT16_C(14299), UINT16_C( 9105), UINT16_C(42075),
                         UINT16_C(43171), UINT16_C(58442), UINT16_C( 4291), UINT16_C(12765)),
      simde_x_vloadq_u16(UINT16_C( 5858), UINT16_C(65395), UINT16_C(61127), UINT16_C(48519),
                         UINT16_C(32167), UINT16_C( 9638), UINT16_C(14112), UINT16_C(20527)),
      simde_x_vloadq_u16(UINT16_C(59636), UINT16_C( 8073), UINT16_C(23342), UINT16_C(54049),
                         UINT16_C( 4375), UINT16_C(13749), UINT16_C(14988), UINT16_C( 5032)),
      simde_x_vloadq_u16(UINT16_C(53346), UINT16_C(22513), UINT16_C(36856), UINT16_C(47605),
                         UINT16_C(15379), UINT16_C(62553), UINT16_C( 5167), UINT16_C(62806)) },
    { simde_x_vloadq_u16(UINT16_C(12030), UINT16_C(63375), UINT16_C(54277), UINT16_C(33757),
                         UINT16_C(39976), UINT16_C(47952), UINT16_C( 7690), UINT16_C(24593)),
      simde_x_vloadq_u16(UINT16_C(50186), UINT16_C(22682), UINT16_C(57446), UINT16_C(49279),
                         UINT16_C(16217), UINT16_C(61348), UINT16_C(29934), UINT16_C( 7867)),
      simde_x_vloadq_u16(UINT16_C(44510), UINT16_C(14773), UINT16_C(51086), UINT16_C(23057),
                         UINT16_C(31622), UINT16_C(27709), UINT16_C(33650), UINT16_C(40090)),
      simde_x_vloadq_u16(UINT16_C( 6354), UINT16_C(55466), UINT16_C(47917), UINT16_C( 7535),
                         UINT16_C(55381), UINT16_C(14313), UINT16_C(11406), UINT16_C(56816)) },
    { simde_x_vloadq_u16(UINT16_C(12501), UINT16_C(38993), UINT16_C(35707), UINT16_C(46252),
                         UINT16_C( 6633), UINT16_C(63188), UINT16_C(56938), UINT16_C( 8282)),
      simde_x_vloadq_u16(UINT16_C(59394), UINT16_C(48320), UINT16_C(49299), UINT16_C(29826),
                         UINT16_C(62454), UINT16_C(42197), UINT16_C(43864), UINT16_C(55919)),
      simde_x_vloadq_u16(UINT16_C(23242), UINT16_C(49905), UINT16_C( 6337), UINT16_C(51670),
                         UINT16_C(44226), UINT16_C(33997), UINT16_C(60972), UINT16_C(18539)),
      simde_x_vloadq_u16(UINT16_C(41885), UINT16_C(40578), UINT16_C(58281), UINT16_C( 2560),
                         UINT16_C(53941), UINT16_C(54988), UINT16_C( 8510), UINT16_C(36438)) },
    { simde_x_vloadq_u16(UINT16_C(12671), UINT16_C(25068), UINT16_C(63714), UINT16_C(49251),
                         UINT16_C(45467), UINT16_C(31481), UINT16_C(27327), UINT16_C(65321)),
      simde_x_vloadq_u16(UINT16_C(65519), UINT16_C(20844), UINT16_C(65087), UINT16_C( 1081),
                         UINT16_C(63573), UINT16_C(13032), UINT16_C(63387), UINT16_C(62770)),
      simde_x_vloadq_u16(UINT16_C(32470), UINT16_C(43864), UINT16_C(27040), UINT16_C(18869),
                         UINT16_C(24074), UINT16_C(36237), UINT16_C(60807), UINT16_C(29942)),
      simde_x_vloadq_u16(UINT16_C(45158), UINT16_C(48088), UINT16_C(25667), UINT16_C( 1503),
                         UINT16_C( 5968), UINT16_C(54686), UINT16_C(24747), UINT16_C(32493)) },
    { simde_x_vloadq_u16(UINT16_C(53609), UINT16_C(30316), UINT16_C( 7799), UINT16_C(64089),
                         UINT16_C(30538), UINT16_C(28343), UINT16_C(24442), UINT16_C(10089)),
      simde_x_vloadq_u16(UINT16_C(61696), UINT16_C(36690), UINT16_C(38533), UINT16_C(62668),
                         UINT16_C(34102), UINT16_C(59501), UINT16_C(16763), UINT16_C(45115)),
      simde_x_vloadq_u16(UINT16_C(40658), UINT16_C(14536), UINT16_C(17811), UINT16_C(62218),
                         UINT16_C(24518), UINT16_C(52278), UINT16_C(60297), UINT16_C(34676)),
      simde_x_vloadq_u16(UINT16_C(32571), UINT16_C( 8162), UINT16_C(52613), UINT16_C(63639),
                         UINT16_C(20954), UINT16_C(21120), UINT16_C( 2440), UINT16_C(65186)) },
    { simde_x_vloadq_u16(UINT16_C(45857), UINT16_C(25708), UINT16_C(41662), UINT16_C(23467),
                         UINT16_C(20436), UINT16_C( 8225), UINT16_C( 5639), UINT16_C(56926)),
      simde_x_vloadq_u16(UINT16_C( 3742), UINT16_C(34917), UINT16_C( 3175), UINT16_C( 2992),
                         UINT16_C(55591), UINT16_C(53596), UINT16_C(50965), UINT16_C(14081)),
      simde_x_vloadq_u16(UINT16_C(  312), UINT16_C(18836), UINT16_C( 7053), UINT16_C(19919),
                         UINT16_C(27594), UINT16_C( 5809), UINT16_C(21545), UINT16_C(11635)),
      simde_x_vloadq_u16(UINT16_C(42427), UINT16_C( 9627), UINT16_C(45540), UINT16_C(40394),
                         UINT16_C(57975), UINT16_C(25974), UINT16_C(41755), UINT16_C(54480)) },
    { simde_x_vloadq_u16(UINT16_C(51599), UINT16_C(42707), UINT16_C(42244), UINT16_C( 4504),
                         UINT16_C(10520), UINT16_C(23283), UINT16_C(60534), UINT16_C(30224)),
      simde_x_vloadq_u16(UINT16_C(15741), UINT16_C(50300), UINT16_C(56571), UINT16_C(57303),
                         UINT16_C(63247), UINT16_C( 1109), UINT16_C(41472), UINT16_C(15767)),
      simde_x_vloadq_u16(UINT16_C(63991), UINT16_C(54984), UINT16_C(59119), UINT16_C( 8970),
                         UINT16_C(56626), UINT16_C(14096), UINT16_C(19265), UINT16_C(21657)),
      simde_x_vloadq_u16(UINT16_C(34313), UINT16_C(47391), UINT16_C(44792), UINT16_C(21707),
                         UINT16_C( 3899), UINT16_C(36270), UINT16_C(38327), UINT16_C(36114)) },
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
    { simde_x_vloadq_u32(UINT32_C(1616609488), UINT32_C( 456211341),
                         UINT32_C(2166919681), UINT32_C( 685112095)),
      simde_x_vloadq_u32(UINT32_C(3177682339), UINT32_C(3424097844),
                         UINT32_C(1989450775), UINT32_C(2689991868)),
      simde_x_vloadq_u32(UINT32_C(3849029281), UINT32_C(1469789209),
                         UINT32_C(4129566631), UINT32_C(3296141425)),
      simde_x_vloadq_u32(UINT32_C(2287956430), UINT32_C(2796870002),
                         UINT32_C(  12068241), UINT32_C(1291261652)) },
    { simde_x_vloadq_u32(UINT32_C( 937164368), UINT32_C(2757436305),
                         UINT32_C(3830098083), UINT32_C( 836571331)),
      simde_x_vloadq_u32(UINT32_C(4285732578), UINT32_C(3179802311),
                         UINT32_C( 631668135), UINT32_C(1345271584)),
      simde_x_vloadq_u32(UINT32_C( 529131764), UINT32_C(3542178606),
                         UINT32_C( 901058839), UINT32_C( 329792140)),
      simde_x_vloadq_u32(UINT32_C(1475530850), UINT32_C(3119812600),
                         UINT32_C(4099488787), UINT32_C(4116059183)) },
    { simde_x_vloadq_u32(UINT32_C(4153356030), UINT32_C(2212353029),
                         UINT32_C(3142622248), UINT32_C(1611734538)),
      simde_x_vloadq_u32(UINT32_C(1486537738), UINT32_C(3229605990),
                         UINT32_C(4020518745), UINT32_C( 515601646)),
      simde_x_vloadq_u32(UINT32_C( 968207838), UINT32_C(1511114638),
                         UINT32_C(1815968646), UINT32_C(2627371890)),
      simde_x_vloadq_u32(UINT32_C(3635026130), UINT32_C( 493861677),
                         UINT32_C( 938072149), UINT32_C(3723504782)) },
    { simde_x_vloadq_u32(UINT32_C(2555457749), UINT32_C(3031206779),
                         UINT32_C(4141095401), UINT32_C( 542826090)),
      simde_x_vloadq_u32(UINT32_C(3166758914), UINT32_C(1954726035),
                         UINT32_C(2765485046), UINT32_C(3664751448)),
      simde_x_vloadq_u32(UINT32_C(3270597322), UINT32_C(3386251457),
                         UINT32_C(2228071618), UINT32_C(1215032876)),
      simde_x_vloadq_u32(UINT32_C(2659296157), UINT32_C( 167764905),
                         UINT32_C(3603681973), UINT32_C(2388074814)) },
    { simde_x_vloadq_u32(UINT32_C(1642869119), UINT32_C(3227777250),
                         UINT32_C(2063184283), UINT32_C(4280904383)),
      simde_x_vloadq_u32(UINT32_C(1366097903), UINT32_C(  70909503),
                         UINT32_C( 854128725), UINT32_C(4113758107)),
      simde_x_vloadq_u32(UINT32_C(2874703574), UINT32_C(1236625824),
                         UINT32_C(2374852106), UINT32_C(1962339719)),
      simde_x_vloadq_u32(UINT32_C(3151474790), UINT32_C(  98526275),
                         UINT32_C(3583907664), UINT32_C(2129485995)) },
    { simde_x_vloadq_u32(UINT32_C(1986842985), UINT32_C(4200144503),
                         UINT32_C(1857517386), UINT32_C( 661217146)),
      simde_x_vloadq_u32(UINT32_C(2404577536), UINT32_C(4107048581),
                         UINT32_C(3899491638), UINT32_C(2956673403)),
      simde_x_vloadq_u32(UINT32_C( 952671954), UINT32_C(4077536659),
                         UINT32_C(3426115526), UINT32_C(2272586633)),
      simde_x_vloadq_u32(UINT32_C( 534937403), UINT32_C(4170632581),
                         UINT32_C(1384141274), UINT32_C(4272097672)) },
    { simde_x_vloadq_u32(UINT32_C(1684845345), UINT32_C(1537974974),
                         UINT32_C( 539054036), UINT32_C(3730707975)),
      simde_x_vloadq_u32(UINT32_C(2288324254), UINT32_C( 196086887),
                         UINT32_C(3512523047), UINT32_C( 922863381)),
      simde_x_vloadq_u32(UINT32_C(1234436408), UINT32_C(1305418637),
                         UINT32_C( 380726218), UINT32_C( 762532905)),
      simde_x_vloadq_u32(UINT32_C( 630957499), UINT32_C(2647306724),
                         UINT32_C(1702224503), UINT32_C(3570377499)) },
    { simde_x_vloadq_u32(UINT32_C(2798897551), UINT32_C( 295216388),
                         UINT32_C(1525885208), UINT32_C(1980820598)),
      simde_x_vloadq_u32(UINT32_C(3296476541), UINT32_C(3755465979),
                         UINT32_C(  72742671), UINT32_C(1033347584)),
      simde_x_vloadq_u32(UINT32_C(3603495415), UINT32_C( 587917039),
                         UINT32_C( 923852082), UINT32_C(1419332417)),
      simde_x_vloadq_u32(UINT32_C(3105916425), UINT32_C(1422634744),
                         UINT32_C(2376994619), UINT32_C(2366805431)) },
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
