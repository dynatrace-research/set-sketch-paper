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



class DummySketch {
    uint64_t state = 0;
public:
    void add(uint64_t l) {
        state += l;
    }

    uint64_t getState() const {
        return state;
    }

};

class DummyConfig {
public:
    DummySketch create() const {
        return DummySketch();
    }

    uint32_t getNumRegisters() const {return 0;}

    std::string getName() const {return "DummyConfig";}
};

template<typename C>
static string getFileName(const C& config, const std::string& aggregationModeDescription) {
    stringstream ss;
    ss << "data/performance_test(";
    appendInfo(ss, config);
    ss << "aggregationMode=" << aggregationModeDescription << ";";
    ss << ").csv";
    return ss.str();
}

static const uint64_t wyhashSecret[4] = {UINT64_C(0xbbc3be7c929be0ca), UINT64_C(0x2cfbaea4f1028efe), UINT64_C(0xc04f8e039a014db9), UINT64_C(0x28b6e9976c77fe03)};

class RandomNumbers {
    uint64_t seed;
    uint64_t mSize;
public:

    class RandomNumberIterator {
        uint64_t seed;
        uint64_t index;
    public:

        RandomNumberIterator(uint64_t seed, uint64_t index): seed(seed), index(index) {}

        uint64_t operator*() const {
            return wyhash(&index, 8, seed, wyhashSecret);
        }

        RandomNumberIterator operator++() {
            index += 1;
            return *this;
        }

        bool operator!=(RandomNumberIterator& it) const {
            return index != it.index;
        }
    };

    RandomNumbers(uint64_t seed, uint64_t size) : seed(seed), mSize(size) {}

    RandomNumberIterator begin() const {
        return RandomNumberIterator(seed, UINT64_C(0));
    }

    RandomNumberIterator end() const {
        return RandomNumberIterator(seed, mSize);
    }

    uint64_t size() const {return mSize;}
};


struct StreamAggregation {

    template<typename S, typename D>
    void aggregate(S& sketch, const D& data) const {
        for(uint64_t d : data) {
            sketch.add(d);
        }
    }

    string getDescription() const {
        return "stream";
    }

};

struct BulkAggregation {

    template<typename S, typename D>
    void aggregate(S& sketch, const D& data) const {
        sketch.addAll(data);
    }

    string getDescription() const {
        return "bulk";
    }

};

template<typename S> uint64_t consumeState(const S& sketch) {
    uint64_t sum = 0;
    for(auto r : sketch.getState()) sum += r;
    return sum;
}

uint64_t consumeState(const DummySketch& sketch) {
    return sketch.getState();
}

template<typename C>
std::vector<uint64_t> getCardinalities(C&& config) {

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
    if (!std::is_same<typename std::remove_reference<C>::type, MinHashConfig>::value) {
        // do not add large cardinalities for minhash, test would take too much time otherwise
        cardinalities.push_back(200000);
        cardinalities.push_back(500000);
        cardinalities.push_back(1000000);
        cardinalities.push_back(2000000);
        cardinalities.push_back(5000000);
        cardinalities.push_back(10000000);
    }

    return cardinalities;
}

template<typename C, typename A>
void test(uint64_t seed, C&& config, const A& aggregationMode) {

    const uint64_t numCycles = 1000; // 1000 takes 1h 45min

    const vector<uint64_t> cardinalities = getCardinalities(config);

    vector<double> recordingTimeSeconds1(cardinalities.size(), 0);
    vector<double> recordingTimeSeconds2(cardinalities.size(), 0);
    vector<uint64_t> sumOfConsumedStates(cardinalities.size(), 0);

    for(uint64_t cardinalityIdx = 0; cardinalityIdx < cardinalities.size(); ++cardinalityIdx) {
        const uint64_t cardinality = cardinalities[cardinalityIdx];
        uint64_t sum = 0;
        auto beginMeasurement1 = chrono::high_resolution_clock::now();
        double sumMeasurement2Seconds = 0;
        for(uint64_t cycleCounter = 0; cycleCounter < numCycles; ++cycleCounter) {
            auto sketch = config.create();
            auto beginMeasurement2 = chrono::high_resolution_clock::now();
            aggregationMode.aggregate(sketch, RandomNumbers(seed, cardinality));
            auto endMeasurement2 = chrono::high_resolution_clock::now();
            sumMeasurement2Seconds += std::chrono::duration<double>(endMeasurement2-beginMeasurement2).count();
            sum += consumeState(sketch);
        }
        auto endMeasurement1 = chrono::high_resolution_clock::now();
        recordingTimeSeconds1[cardinalityIdx] =  std::chrono::duration<double>(endMeasurement1-beginMeasurement1).count() / static_cast<double>(numCycles);
        recordingTimeSeconds2[cardinalityIdx] = sumMeasurement2Seconds / static_cast<double>(numCycles);
        sumOfConsumedStates[cardinalityIdx] = sum;
    }

    ofstream f(getFileName(config, aggregationMode.getDescription()));
    appendInfo(f, config);
    f << "numCycles=" << numCycles << ";";
    f << "aggregationMode=" << aggregationMode.getDescription() << ";";
    f << endl;
    f << "cardinality; avg time in seconds (incl. allocation); avg time in seconds (excl. allocation);sum of consumed states;" << endl;
    for(uint64_t cardinalityIdx = 0; cardinalityIdx < cardinalities.size(); ++cardinalityIdx) {
        f << cardinalities[cardinalityIdx] << ";" << recordingTimeSeconds1[cardinalityIdx] << ";" << recordingTimeSeconds2[cardinalityIdx] << ";" << sumOfConsumedStates[cardinalityIdx] << ";" << endl;
    }
    f.close();
}

void appendInfo(std::ostream& os, const DummyConfig& config)
{
    os << "dummy;";
}

int main(int argc, char* argv[]) {

    mt19937_64 dataSeedRng(UINT64_C(0x291be5007a3d06fc));

    test(dataSeedRng(), DummyConfig(), StreamAggregation());

    std::vector<uint64_t> registerSizeExponents = {8, 12};

    for(uint64_t registerSizeExponent : registerSizeExponents) {

        const uint32_t numRegisters = UINT32_C(1) << registerSizeExponent;

        test(dataSeedRng(), HyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(registerSizeExponent, 64 - registerSizeExponent), StreamAggregation());
        test(dataSeedRng(), HyperLogLogConfig<Registers<uint8_t>>(registerSizeExponent, 64 - registerSizeExponent), StreamAggregation());
        
        test(dataSeedRng(), MinHashConfig(numRegisters), StreamAggregation());
        
        test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(numRegisters, 2., 62), StreamAggregation());
        test(dataSeedRng(), GeneralizedHyperLogLogConfig<Registers<uint8_t>>(numRegisters, 2., 62), StreamAggregation());
        
        test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint8_t>>(numRegisters, 2., 20, 62), StreamAggregation());
        test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint8_t>>(numRegisters, 2., 20, 62), StreamAggregation());
        test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint8_t>>(numRegisters, 2., 20, 62), BulkAggregation());
        test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint8_t>>(numRegisters, 2., 20, 62), BulkAggregation());

        test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(numRegisters, 1.001, 62), StreamAggregation());
        test(dataSeedRng(), GeneralizedHyperLogLogConfig<Registers<uint8_t>>(numRegisters, 1.001, 62), StreamAggregation());
        
        test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint8_t>>(numRegisters, 1.001, 20, 62), StreamAggregation());
        test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint8_t>>(numRegisters, 1.001, 20, 62), StreamAggregation());
        test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint8_t>>(numRegisters, 1.001, 20, 62), BulkAggregation());
        test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint8_t>>(numRegisters, 1.001, 20, 62), BulkAggregation());
    }
}