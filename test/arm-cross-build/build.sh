#!/bin/sh
export BUILD_TYPE="Coverage"
export BUILD_CPP_TESTS=ON
export CMAKE_GENERATOR='Ninja'
export RUN_TESTS=true
export OPTIMIZATION_FLAGS=''
export DIAGNOSTIC_FLAGS='-Wall -Wextra -Werror'
export CMAKE_C_COMPILER=arm-linux-gnueabihf-gcc
export CMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++
export C_COMPILER=arm-linux-gnueabihf-gcc
export CXX_COMPILER=arm-linux-gnueabihf-g++
export CC=arm-linux-gnueabihf-gcc
export CXX=arm-linux-gnueabihf-g++
export ARCH_FLAGS="-march=armv8-a"

${CONFIGURE_WRAPPER} cmake .. \
        -G "${CMAKE_GENERATOR}" \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DBUILD_CPP_TESTS=${BUILD_CPP_TESTS} \
        -DCMAKE_CROSSCOMPILING_EMULATOR="${TEST_WRAPPER}" \
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
        -DCMAKE_C_FLAGS="${ARCH_FLAGS} ${OPTIMIZATION_FLAGS} ${DIAGNOSTIC_FLAGS} ${COMPILER_FLAGS} ${CFLAGS}" \
        -DCMAKE_CXX_FLAGS="${ARCH_FLAGS} ${OPTIMIZATION_FLAGS} ${DIAGNOSTIC_FLAGS} ${COMPILER_FLAGS} ${CXXFLAGS}" \
        ${CMAKE_ARGS} || (cat CMakeFiles/CMakeError.log && false) && \
        ${BUILD_WRAPPER} cmake --build .


