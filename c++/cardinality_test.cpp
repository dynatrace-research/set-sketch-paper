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

#include <random>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

template<typename C>
static string getFileName(const C& config) {
    stringstream ss;
    ss << "data/cardinality_test(";
    appendInfo(ss, config);
    ss << ").csv";
    return ss.str();
}

template<typename C>
void test(uint64_t seed, const C& config) {

    const uint64_t numExamples = 1000; // takes approx. 1h 30min

    const vector<uint64_t> cardinalities = getCardinalities(10000000, 0.01);

    const uint64_t seedSize = 256;

    vector<vector<double>> simpleEstimates(cardinalities.size(), vector<double>(numExamples));
    vector<vector<double>> simpleCorrectedEstimates(cardinalities.size(), vector<double>(numExamples));
    vector<vector<double>> mlEstimates(cardinalities.size(), vector<double>(numExamples));
    vector<vector<double>> mlCorrectedEstimates(cardinalities.size(), vector<double>(numExamples));

    mt19937 initialRng(seed);
    vector<uint32_t> seeds(numExamples * seedSize);
    generate(seeds.begin(), seeds.end(), initialRng);

    #pragma omp parallel for
    for (uint64_t i = 0; i < numExamples; ++i) {
        seed_seq seedSequence(seeds.begin() + i * seedSize, seeds.begin() + (i + 1) * seedSize);
        mt19937_64 rng(seedSequence);
        uint64_t trueCardinality = 0;
        auto sketch = config.create();
        for(uint64_t cardinalityIdx = 0; cardinalityIdx < cardinalities.size(); ++cardinalityIdx) {
            const uint64_t testCardinality = cardinalities[cardinalityIdx];
            while(trueCardinality < testCardinality) {
                sketch.add(rng());
                trueCardinality += 1;
            }
            simpleEstimates[cardinalityIdx][i] = sketch.estimateCardinalitySimple(false);
            simpleCorrectedEstimates[cardinalityIdx][i] = sketch.estimateCardinalitySimple(true);
            mlEstimates[cardinalityIdx][i] = sketch.estimateCardinalityML(false);
            mlCorrectedEstimates[cardinalityIdx][i] = sketch.estimateCardinalityML(true);
        }
    }

    ofstream f(getFileName(config));
    appendInfo(f, config);
    f << "numExamples=" << numExamples << ";";
    f << endl;
    f << "true cardinality; simple mean; simple mse; simple corrected mean; simple corrected mse; ml mean; ml mse;ml corrected mean; ml corrected mse;" << endl;
    for(uint64_t cardinalityIdx = 0; cardinalityIdx < cardinalities.size(); ++cardinalityIdx) {
        uint64_t trueCardinality = cardinalities[cardinalityIdx];
        double meanSimple = calculateMean(simpleEstimates[cardinalityIdx]);
        double mseSimple = calculateMSE(simpleEstimates[cardinalityIdx], trueCardinality);
        double meanSimpleCorrected = calculateMean(simpleCorrectedEstimates[cardinalityIdx]);
        double mseSimpleCorrected = calculateMSE(simpleCorrectedEstimates[cardinalityIdx], trueCardinality);
        double meanML = calculateMean(mlEstimates[cardinalityIdx]);
        double mseML = calculateMSE(mlEstimates[cardinalityIdx], trueCardinality);
        double meanMLCorrected = calculateMean(mlCorrectedEstimates[cardinalityIdx]);
        double mseMLCorrected = calculateMSE(mlCorrectedEstimates[cardinalityIdx], trueCardinality);

        f << setprecision(numeric_limits< double >::max_digits10) << scientific <<
            trueCardinality << ";" <<
            meanSimple << "; " << mseSimple << "; " <<
            meanSimpleCorrected << "; " << mseSimpleCorrected << "; " <<
            meanML << "; " << mseML << "; " <<
            meanMLCorrected << "; " << mseMLCorrected << "; " <<
            endl;
    }

    f.close();

}

int main(int argc, char* argv[]) {

    mt19937_64 dataSeedRng(0xf5c1f864cefbf048);
    mt19937_64 configSeedRng(0x79b5d54d2f1489da);

    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(256, 2., 62, configSeedRng()));
    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(1024, 2., 62, configSeedRng()));
    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(4096, 2., 62, configSeedRng()));
    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(16384, 2., 62, configSeedRng()));

    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint8_t>>(256, 2., 30, 62, configSeedRng()));
    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint8_t>>(1024, 2., 30, 62, configSeedRng()));
    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint8_t>>(4096, 2., 30, 62, configSeedRng()));
    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint8_t>>(16384, 2., 30, 62, configSeedRng()));

    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint8_t>>(256, 2., 30, 62, configSeedRng()));
    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint8_t>>(1024, 2., 30, 62, configSeedRng()));
    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint8_t>>(4096, 2., 30, 62, configSeedRng()));
    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint8_t>>(16384, 2., 30, 62, configSeedRng()));

    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint16_t>>(256, 1.001, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()));
    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint16_t>>(1024, 1.001, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()));
    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint16_t>>(4096, 1.001, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()));
    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint16_t>>(16384, 1.001, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()));

    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint16_t>>(256, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()));
    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint16_t>>(1024, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()));
    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint16_t>>(4096, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()));
    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint16_t>>(16384, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()));

    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint16_t>>(256, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()));
    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint16_t>>(1024, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()));
    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint16_t>>(4096, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()));
    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint16_t>>(16384, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()));

}