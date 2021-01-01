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

#include "sketch.hpp"
#include "util.hpp"
#include "wyhash/wyhash.h"

#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

using namespace std;

template<typename C>
static string getFileName(const C& config) {
    stringstream ss;
    ss << "data/performance_test(";
    appendInfo(ss, config);
    ss << ").csv";
    return ss.str();
}

template<typename C>
void test(uint64_t seed, C&& config, const vector<uint64_t>& cardinalities) {

    const uint64_t numCycles = 1000; // takes approx. 210 min

    uint64_t state = seed;
    
    vector<double> recordingTimeSeconds(cardinalities.size(), 0);
    vector<double> estimates(cardinalities.size(), 0);

    for(uint64_t cardinalityIdx = 0; cardinalityIdx < cardinalities.size(); ++cardinalityIdx) {
        const uint64_t cardinality = cardinalities[cardinalityIdx];
        auto beginMeasurement = chrono::high_resolution_clock::now();    
        double e = 0;
        for(uint64_t cycleCounter = 0; cycleCounter < numCycles; ++cycleCounter) {
            auto sketch = config.create();
            for(uint64_t j = 0; j < cardinality; ++j) sketch.add(wyrand(&state));
            e += sketch.estimateCardinalitySimple(false);
        }
        auto endMeasurement = chrono::high_resolution_clock::now();    

        recordingTimeSeconds[cardinalityIdx] = static_cast<double>(chrono::duration_cast<chrono::nanoseconds>(endMeasurement-beginMeasurement).count()) / (1e9 * numCycles);
        estimates[cardinalityIdx] = e / numCycles;
    }

    ofstream f(getFileName(config));
    appendInfo(f, config);
    f << "numCycles=" << numCycles << ";";
    f << endl;
    f << "cardinality; avg time in seconds;avg estimated cardinality;" << endl;
    for(uint64_t cardinalityIdx = 0; cardinalityIdx < cardinalities.size(); ++cardinalityIdx) {
        f << cardinalities[cardinalityIdx] << ";" << recordingTimeSeconds[cardinalityIdx] << ";" << estimates[cardinalityIdx] << endl;
    }
    f.close();
}

class DummySketch {
    uint64_t state = 0;
public:
    void add(uint64_t l) {
        state += l;
    }

    double estimateCardinalitySimple(bool b) const {
        return state;
    }   
};


class DummyConfig {
public:
    DummySketch create() const {
        return DummySketch();
    }
};

void appendInfo(std::ostream& os, const DummyConfig& config)
{
    os << "dummy;";
}

std::vector<uint64_t> getCardinalities() {
    vector<uint64_t> cardinalities;
    cardinalities.push_back(1);
    cardinalities.push_back(2);
    cardinalities.push_back(5);
    cardinalities.push_back(10);
    cardinalities.push_back(20);
    cardinalities.push_back(50);
    cardinalities.push_back(100);
    cardinalities.push_back(200);
    cardinalities.push_back(500);
    cardinalities.push_back(1000);
    cardinalities.push_back(2000);
    cardinalities.push_back(5000);
    cardinalities.push_back(10000);
    cardinalities.push_back(20000);
    cardinalities.push_back(50000);
    cardinalities.push_back(100000);
    cardinalities.push_back(200000);
    cardinalities.push_back(500000);
    cardinalities.push_back(1000000);
    cardinalities.push_back(2000000);
    cardinalities.push_back(5000000);
    cardinalities.push_back(10000000);
    return cardinalities;
}

int main(int argc, char* argv[]) {

    mt19937_64 configSeedRng(UINT64_C(0x9799d8d65c8534be));

    mt19937_64 dataSeedRng(UINT64_C(0x291be5007a3d06fc));
    const vector<uint64_t> cardinalities = getCardinalities();

    test(dataSeedRng(), DummyConfig(), cardinalities);

    test(dataSeedRng(), HyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(8, 56), cardinalities);
    test(dataSeedRng(), HyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(10, 54), cardinalities);
    test(dataSeedRng(), HyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(12, 52), cardinalities);
    test(dataSeedRng(), HyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(14, 50), cardinalities);

    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(256, 2., 62, configSeedRng()), cardinalities);
    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(1024, 2., 62, configSeedRng()), cardinalities);
    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(4096, 2., 62, configSeedRng()), cardinalities);
    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(16384, 2., 62, configSeedRng()), cardinalities);

    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint8_t>>(256, 2., 30, 62, configSeedRng()), cardinalities);
    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint8_t>>(1024, 2., 30, 62, configSeedRng()), cardinalities);
    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint8_t>>(4096, 2., 30, 62, configSeedRng()), cardinalities);
    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint8_t>>(16384, 2., 30, 62, configSeedRng()), cardinalities);

    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint8_t>>(256, 2., 30, 62, configSeedRng()), cardinalities);
    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint8_t>>(1024, 2., 30, 62, configSeedRng()), cardinalities);
    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint8_t>>(4096, 2., 30, 62, configSeedRng()), cardinalities);
    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint8_t>>(16384, 2., 30, 62, configSeedRng()), cardinalities);

    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint16_t>>(256, 1.001, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()), cardinalities);
    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint16_t>>(1024, 1.001, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()), cardinalities);
    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint16_t>>(4096, 1.001, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()), cardinalities);
    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint16_t>>(16384, 1.001, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()), cardinalities);

    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint16_t>>(256, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()), cardinalities);
    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint16_t>>(1024, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()), cardinalities);
    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint16_t>>(4096, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()), cardinalities);
    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint16_t>>(16384, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()), cardinalities);

    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint16_t>>(256, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()), cardinalities);
    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint16_t>>(1024, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()), cardinalities);
    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint16_t>>(4096, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()), cardinalities);
    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint16_t>>(16384, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()), cardinalities);

}