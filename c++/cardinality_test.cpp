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

    const uint64_t numExamples = 10000; // 10000 takes 1h 40min

    const vector<uint64_t> cardinalities = getCardinalities(10000000, 0.01);

    const uint64_t seedSize = 256;

    const vector<string> estimatorLabels = config.getEstimator().getCardinalityEstimateLabels();
    const size_t numEstimates = estimatorLabels.size();
    vector<vector<vector<double>>> estimates(numEstimates, vector<vector<double>>(cardinalities.size(), vector<double>(numExamples)));

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
            const auto results = config.getEstimator().estimateCardinalities(sketch.getState());
            for(size_t j = 0; j < numEstimates; ++j) estimates[j][cardinalityIdx][i] = results[j];
        }
    }

    ofstream f(getFileName(config));
    appendInfo(f, config);
    f << "numExamples=" << numExamples << ";";
    f << endl;
    f << "true cardinality; ";
    for(size_t j = 0; j < numEstimates; ++j) f << estimatorLabels[j] << " mean; " << estimatorLabels[j] << " mse; " << estimatorLabels[j] << " standard deviation; " << estimatorLabels[j] << " kurtosis; ";
    f << endl;
    for(uint64_t cardinalityIdx = 0; cardinalityIdx < cardinalities.size(); ++cardinalityIdx) {
        uint64_t trueCardinality = cardinalities[cardinalityIdx];
        f << setprecision(numeric_limits< double >::max_digits10) << scientific << trueCardinality << "; ";
        for(size_t j = 0; j < numEstimates; ++j) {
            double mean = calculateMean(estimates[j][cardinalityIdx]);
            double mse = calculateMSE(estimates[j][cardinalityIdx], trueCardinality);
            double standardDeviation = calculateStandardDeviation(estimates[j][cardinalityIdx]);
            double kurtosis = calculateKurtosis(estimates[j][cardinalityIdx]);
            f << mean << "; " << mse << "; " << standardDeviation << "; " << kurtosis << "; ";
        }
        f << endl;
    }

    f.close();

}

int main(int argc, char* argv[]) {

    mt19937_64 dataSeedRng(0xf5c1f864cefbf048);

    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(256, 2., 62));
    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(4096, 2., 62));

    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint8_t>>(256, 2., 20, 62));
    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint8_t>>(4096, 2., 20, 62));

    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint8_t>>(256, 2., 20, 62));
    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint8_t>>(4096, 2., 20, 62));

    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint16_t>>(256, 1.001, std::numeric_limits<uint16_t>::max() - 1));
    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint16_t>>(4096, 1.001, std::numeric_limits<uint16_t>::max() - 1));

    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint16_t>>(256, 1.001, 20, std::numeric_limits<uint16_t>::max() - 1));
    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint16_t>>(4096, 1.001, 20, std::numeric_limits<uint16_t>::max() - 1));

    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint16_t>>(256, 1.001, 20, std::numeric_limits<uint16_t>::max() - 1));
    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint16_t>>(4096, 1.001, 20, std::numeric_limits<uint16_t>::max() - 1));

}