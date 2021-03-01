// Copyright (c) 2012-2021 Dynatrace LLC. All rights reserved.
//
// This software and associated documentation files (the "Software")
// are being made available by Dynatrace LLC for purposes of
// illustrating the implementation of certain algorithms which have
// been published by Dynatrace LLC. Permission is hereby granted,
// free of charge, to any person obtaining a copy of the Software,
// to view and use the Software for internal, non-productive,
// non-commercial purposes only – the Software may not be used to
// process live data or distributed, sublicensed, modified and/or
// sold either alone or as part of or in combination with any other
// software.
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#include "bitstream_random.hpp"

#include <random>

using namespace std;


int main(int argc, char* argv[]) {

    mt19937_64 rng(UINT64_C(0x356fc7675f6cce28));

    uniform_int_distribution<uint8_t> dist(1, 64);

    WyrandBitStream s1(UINT64_C(0xc2881c5d6c802af7));
    WyrandBitStream s2(UINT64_C(0xc2881c5d6c802af7));

    uint64_t numIterations = 1000000;

    for(uint64_t i = 0; i < numIterations; ++i) {

        uint8_t numBits = dist(rng);

        uint64_t v1 = s1(numBits);
        uint64_t v2 = 0;
        for (uint8_t k = 0; k < numBits; ++k) {
            v2 <<= 1;
            v2 |= s2();
        }

        assert(v1 == v2);

    }

}