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
#include <type_traits>
#include <chrono>
#include <set>

using namespace std;

template<typename C>
static string getFileName(const C& config) {
    stringstream ss;
    ss << "data/joint_test(";
    appendInfo(ss, config);
    ss << ").csv";
    return ss.str();
}

static vector<tuple<uint64_t, uint64_t, uint64_t>> getCardinalityTuples() {

    uint64_t unionCardinality1 = 1000000;
    uint64_t unionCardinality2 = 1000;
    const vector<uint64_t> intersections1 = {100000, 10000, 1000};
    const vector<uint64_t> intersections2 = {100, 10, 1};
    const double ratioFactor = 1.5;
    const double maxRatio = 1000;

    set<tuple<uint64_t, uint64_t, uint64_t>> cardinalityTuples;

    for(uint64_t intersection : intersections1) {
        
        double ratio = 1;
        while(true) {
            
            uint64_t diff1 = static_cast<uint64_t>(floor((unionCardinality1 - intersection) / (1 + 1./ratio))); 
            uint64_t diff2 = unionCardinality1 - intersection - diff1;

            assert(diff1 >= 0);
            assert(diff2 >= 0);

            cardinalityTuples.emplace(make_tuple(diff1, diff2, intersection));

            double trueRatio = static_cast<double>(diff1) / static_cast<double>(diff2);

            if(trueRatio >= maxRatio) break;

            ratio *= ratioFactor;
        }
    }

    for(uint64_t intersection : intersections2) {
        
        double ratio = 1;
        while(true) {
            
            uint64_t diff1 = static_cast<uint64_t>(floor((unionCardinality2 - intersection) / (1 + 1./ratio))); 
            uint64_t diff2 = unionCardinality2 - intersection - diff1;

            assert(diff1 >= 0);
            assert(diff2 >= 0);

            cardinalityTuples.emplace(make_tuple(diff1, diff2, intersection));

            double trueRatio = static_cast<double>(diff1) / static_cast<double>(diff2);

            if(trueRatio >= maxRatio) break;

            ratio *= ratioFactor;
        }
    }

    return vector<tuple<uint64_t, uint64_t, uint64_t>>(cardinalityTuples.begin(), cardinalityTuples.end());
}

template<typename C, typename S>
static S composeSketch(const C& config, uint64_t trueCardinality, const vector<S>& sketches) {
    S result(config);
    for(int8_t k = 63; k >= 0; k -= 1){
        if ((trueCardinality & (UINT64_C(1) << k)) != UINT64_C(0)) {
            assert(static_cast<uint64_t>(k) < sketches.size());
            result.merge(sketches[k]);
        }   
    }
    return result;
}

template<typename C>
void test(uint64_t seed, const C& config, bool rangeCorrection) {

    const uint64_t seedSize = 256;

    const uint64_t numExamples = 1000; // takes approx. 1h

    typedef typename C::SketchType sketch_type; 

    mt19937 initialRng(seed);
    vector<uint32_t> seeds(numExamples * seedSize);
    generate(seeds.begin(), seeds.end(), initialRng);

    const auto cardinalityTuples = getCardinalityTuples();

    uint64_t maxCardinalityA = 0;
    uint64_t maxCardinalityB = 0;
    uint64_t maxCardinalityX = 0;
    for(auto c :cardinalityTuples) {
        maxCardinalityA = max(maxCardinalityA, get<0>(c));
        maxCardinalityB = max(maxCardinalityB, get<1>(c));
        maxCardinalityX = max(maxCardinalityX, get<2>(c));
    }

    vector<vector<JointEstimationResult>> estJointInclExcl(numExamples);
    vector<vector<JointEstimationResult>> estJointML(numExamples);
    vector<vector<JointEstimationResult>> estJointSimple(numExamples);
    vector<vector<pair<double,double>>> estJaccardColl(numExamples);

    uint64_t nanoSecondsForGeneration = 0;
    uint64_t nanoSecondsForEstimation = 0;

    #pragma omp parallel for
    for (uint64_t i = 0; i < numExamples; ++i) {
        auto beginGeneration = chrono::high_resolution_clock::now();
        seed_seq seedSequence(seeds.begin() + i * seedSize, seeds.begin() + (i + 1) * seedSize);
        mt19937_64 rng(seedSequence);

        vector<sketch_type> sketchesA;
        vector<sketch_type> sketchesB;
        vector<sketch_type> sketchesX;

        for(uint64_t nextCardinality = 1; nextCardinality <= maxCardinalityA; nextCardinality *= 2) {
            sketch_type sketch = config.create();
            for(uint64_t trueCardinality = 0; trueCardinality < nextCardinality; trueCardinality += 1) {
                sketch.add(rng());
            }
            sketchesA.emplace_back(sketch);
        }
        for(uint64_t nextCardinality = 1; nextCardinality <= maxCardinalityB; nextCardinality *= 2) {
            sketch_type sketch = config.create();
            for(uint64_t trueCardinality = 0; trueCardinality < nextCardinality; trueCardinality += 1) {
                sketch.add(rng());
            }
            sketchesB.emplace_back(sketch);
        }
        for(uint64_t nextCardinality = 1; nextCardinality <= maxCardinalityX; nextCardinality *= 2) {
            sketch_type sketch = config.create();
            for(uint64_t trueCardinality = 0; trueCardinality < nextCardinality; trueCardinality += 1) {
                sketch.add(rng());
            }
            sketchesX.emplace_back(sketch);
        }
        auto endGeneration = chrono::high_resolution_clock::now();
        auto generationNanoSeconds = chrono::duration_cast<chrono::nanoseconds>(endGeneration-beginGeneration).count();
        
        #pragma omp atomic
        nanoSecondsForGeneration += generationNanoSeconds;

        auto beginEstimation = chrono::high_resolution_clock::now();
        for (uint64_t j = 0; j < cardinalityTuples.size(); ++j) {

            const auto& cardinalityTuple = cardinalityTuples[j];

            uint64_t trueCardinalityA = get<0>(cardinalityTuple);
            uint64_t trueCardinalityB = get<1>(cardinalityTuple);
            uint64_t trueCardinalityX = get<2>(cardinalityTuple);
            
            sketch_type sketch1 = composeSketch(config, trueCardinalityA, sketchesA);
            sketch_type sketch2 = composeSketch(config, trueCardinalityB, sketchesB);
            sketch_type sketchX = composeSketch(config, trueCardinalityX, sketchesX);
            sketch_type sketch1X = merge(sketch1, sketchX);
            sketch_type sketch2X = merge(sketch2, sketchX);
            estJointML[i].emplace_back(estimateJointML(sketch1X, sketch2X, rangeCorrection));
            estJointSimple[i].emplace_back(estimateJointSimple(sketch1X, sketch2X, rangeCorrection));
            estJointInclExcl[i].emplace_back(estimateJointInclExcl(sketch1X, sketch2X, rangeCorrection));
            estJaccardColl[i].emplace_back(estimateJaccardSimilarityUsingEqualRegisters(sketch1X, sketch2X));
        }
        auto endEstimation = chrono::high_resolution_clock::now();
        auto estimationNanoSeconds = chrono::duration_cast<chrono::nanoseconds>(endEstimation-beginEstimation).count();
        
        #pragma omp atomic
        nanoSecondsForEstimation += estimationNanoSeconds;
    }

    cout << "###########" << endl;
    appendInfo(cout, config);
    cout << "numExamples=" << numExamples << ";";
    cout << endl;
    cout << "seconds for generation = " << (nanoSecondsForGeneration*1e-9) << endl;
    cout << "seconds for estimation = " << (nanoSecondsForEstimation*1e-9) << endl;

    ofstream f(getFileName(config));

    appendInfo(f, config);
    f << "numExamples=" << numExamples << ";";
    f << endl;

    f << "trueDifference1" << ";";
    f << "trueDifference2" << ";";
    f << "trueIntersection" << ";";
    f << "true1" << ";";
    f << "true2" << ";";
    f << "trueUnion" << ";";
    f << "trueJaccard" << ";";
    f << "trueCosine" << ";";
    f << "trueInclusionCoefficient1" << ";";
    f << "trueInclusionCoefficient2" << ";";
    f << "trueAlpha" << ";";
    f << "trueBeta" << ";";

    f << "mlMeanDifference1" << ";";
    f << "mlMeanDifference2" << ";";
    f << "mlMeanIntersection" << ";";
    f << "mlMean1" << ";";
    f << "mlMean2" << ";";
    f << "mlMeanUnion" << ";";
    f << "mlMeanJaccard" << ";";
    f << "mlMeanCosine" << ";";
    f << "mlMeanInclusionCoefficient1" << ";";
    f << "mlMeanInclusionCoefficient2" << ";";
    f << "mlMeanAlpha" << ";";
    f << "mlMeanBeta" << ";";

    f << "mlMSEDifference1" << ";";
    f << "mlMSEDifference2" << ";";
    f << "mlMSEIntersection" << ";";
    f << "mlMSE1" << ";";
    f << "mlMSE2" << ";";
    f << "mlMSEUnion" << ";";
    f << "mlMSEJaccard" << ";";
    f << "mlMSECosine" << ";";
    f << "mlMSEInclusionCoefficient1" << ";";
    f << "mlMSEInclusionCoefficient2" << ";";
    f << "mlMSEAlpha" << ";";
    f << "mlMSEBeta" << ";";

    f << "simpleMeanDifference1" << ";";
    f << "simpleMeanDifference2" << ";";
    f << "simpleMeanIntersection" << ";";
    f << "simpleMean1" << ";";
    f << "simpleMean2" << ";";
    f << "simpleMeanUnion" << ";";
    f << "simpleMeanJaccard" << ";";
    f << "simpleMeanCosine" << ";";
    f << "simpleMeanInclusionCoefficient1" << ";";
    f << "simpleMeanInclusionCoefficient2" << ";";
    f << "simpleMeanAlpha" << ";";
    f << "simpleMeanBeta" << ";";

    f << "simpleMSEDifference1" << ";";
    f << "simpleMSEDifference2" << ";";
    f << "simpleMSEIntersection" << ";";
    f << "simpleMSE1" << ";";
    f << "simpleMSE2" << ";";
    f << "simpleMSEUnion" << ";";
    f << "simpleMSEJaccard" << ";";
    f << "simpleMSECosine" << ";";
    f << "simpleMSEInclusionCoefficient1" << ";";
    f << "simpleMSEInclusionCoefficient2" << ";";
    f << "simpleMSEAlpha" << ";";
    f << "simpleMSEBeta" << ";";

    f << "inclExclMeanDifference1" << ";";
    f << "inclExclMeanDifference2" << ";";
    f << "inclExclMeanIntersection" << ";";
    f << "inclExclMean1" << ";";
    f << "inclExclMean2" << ";";
    f << "inclExclMeanUnion" << ";";
    f << "inclExclMeanJaccard" << ";";
    f << "inclExclMeanCosine" << ";";
    f << "inclExclMeanInclusionCoefficient1" << ";";
    f << "inclExclMeanInclusionCoefficient2" << ";";
    f << "inclExclMeanAlpha" << ";";
    f << "inclExclMeanBeta" << ";";

    f << "inclExclMSEDifference1" << ";";
    f << "inclExclMSEDifference2" << ";";
    f << "inclExclMSEIntersection" << ";";
    f << "inclExclMSE1" << ";";
    f << "inclExclMSE2" << ";";
    f << "inclExclMSEUnion" << ";";
    f << "inclExclMSEJaccard" << ";";
    f << "inclExclMSECosine" << ";";
    f << "inclExclMSEInclusionCoefficient1" << ";";
    f << "inclExclMSEInclusionCoefficient2" << ";";
    f << "inclExclMSEAlpha" << ";";
    f << "inclExclMSEBeta" << ";";

    f << "collLowerBoundMeanJaccard" << ";";
    f << "collUpperBoundMeanJaccard" << ";";
    f << "collLowerBoundMSEJaccard" << ";";
    f << "collUpperBoundMSEJaccard" << ";";

    f << endl;

    for (uint64_t j = 0; j < cardinalityTuples.size(); ++j) {

        const auto& cardinalityTuple = cardinalityTuples[j];
        
        uint64_t trueCardinalityA = get<0>(cardinalityTuple);
        uint64_t trueCardinalityB = get<1>(cardinalityTuple);
        uint64_t trueCardinalityX = get<2>(cardinalityTuple);
        
        JointEstimationResult trueJoint(trueCardinalityA, trueCardinalityB, trueCardinalityX);

        f << trueJoint.getDifference1() << ";";
        f << trueJoint.getDifference2() << ";";
        f << trueJoint.getIntersection() << ";";
        f << trueJoint.get1() << ";";
        f << trueJoint.get2() << ";";
        f << trueJoint.getUnion() << ";";
        f << trueJoint.getJaccard() << ";";
        f << trueJoint.getCosine() << ";";
        f << trueJoint.getInclusionCoefficient1() << ";";
        f << trueJoint.getInclusionCoefficient2() << ";";
        f << trueJoint.getAlpha() << ";";
        f << trueJoint.getBeta() << ";";

        f << calculateMean([&](uint64_t idx){return estJointML[idx][j].getDifference1();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointML[idx][j].getDifference2();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointML[idx][j].getIntersection();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointML[idx][j].get1();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointML[idx][j].get2();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointML[idx][j].getUnion();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointML[idx][j].getJaccard();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointML[idx][j].getCosine();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointML[idx][j].getInclusionCoefficient1();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointML[idx][j].getInclusionCoefficient2();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointML[idx][j].getAlpha();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointML[idx][j].getBeta();}, numExamples) << ";";
        
        f << calculateMSE([&](uint64_t idx){return estJointML[idx][j].getDifference1();}, numExamples, trueJoint.getDifference1()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointML[idx][j].getDifference2();}, numExamples, trueJoint.getDifference2()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointML[idx][j].getIntersection();}, numExamples, trueJoint.getIntersection()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointML[idx][j].get1();}, numExamples, trueJoint.get1()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointML[idx][j].get2();}, numExamples, trueJoint.get2()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointML[idx][j].getUnion();}, numExamples, trueJoint.getUnion()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointML[idx][j].getJaccard();}, numExamples, trueJoint.getJaccard()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointML[idx][j].getCosine();}, numExamples, trueJoint.getCosine()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointML[idx][j].getInclusionCoefficient1();}, numExamples, trueJoint.getInclusionCoefficient1()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointML[idx][j].getInclusionCoefficient2();}, numExamples, trueJoint.getInclusionCoefficient2()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointML[idx][j].getAlpha();}, numExamples, trueJoint.getAlpha()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointML[idx][j].getBeta();}, numExamples, trueJoint.getBeta()) << ";";

        f << calculateMean([&](uint64_t idx){return estJointSimple[idx][j].getDifference1();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointSimple[idx][j].getDifference2();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointSimple[idx][j].getIntersection();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointSimple[idx][j].get1();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointSimple[idx][j].get2();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointSimple[idx][j].getUnion();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointSimple[idx][j].getJaccard();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointSimple[idx][j].getCosine();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointSimple[idx][j].getInclusionCoefficient1();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointSimple[idx][j].getInclusionCoefficient2();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointSimple[idx][j].getAlpha();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointSimple[idx][j].getBeta();}, numExamples) << ";";
        
        f << calculateMSE([&](uint64_t idx){return estJointSimple[idx][j].getDifference1();}, numExamples, trueJoint.getDifference1()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointSimple[idx][j].getDifference2();}, numExamples, trueJoint.getDifference2()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointSimple[idx][j].getIntersection();}, numExamples, trueJoint.getIntersection()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointSimple[idx][j].get1();}, numExamples, trueJoint.get1()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointSimple[idx][j].get2();}, numExamples, trueJoint.get2()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointSimple[idx][j].getUnion();}, numExamples, trueJoint.getUnion()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointSimple[idx][j].getJaccard();}, numExamples, trueJoint.getJaccard()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointSimple[idx][j].getCosine();}, numExamples, trueJoint.getCosine()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointSimple[idx][j].getInclusionCoefficient1();}, numExamples, trueJoint.getInclusionCoefficient1()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointSimple[idx][j].getInclusionCoefficient2();}, numExamples, trueJoint.getInclusionCoefficient2()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointSimple[idx][j].getAlpha();}, numExamples, trueJoint.getAlpha()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointSimple[idx][j].getBeta();}, numExamples, trueJoint.getBeta()) << ";";

        f << calculateMean([&](uint64_t idx){return estJointInclExcl[idx][j].getDifference1();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointInclExcl[idx][j].getDifference2();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointInclExcl[idx][j].getIntersection();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointInclExcl[idx][j].get1();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointInclExcl[idx][j].get2();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointInclExcl[idx][j].getUnion();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointInclExcl[idx][j].getJaccard();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointInclExcl[idx][j].getCosine();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointInclExcl[idx][j].getInclusionCoefficient1();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointInclExcl[idx][j].getInclusionCoefficient2();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointInclExcl[idx][j].getAlpha();}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJointInclExcl[idx][j].getBeta();}, numExamples) << ";";
    
        f << calculateMSE([&](uint64_t idx){return estJointInclExcl[idx][j].getDifference1();}, numExamples, trueJoint.getDifference1()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointInclExcl[idx][j].getDifference2();}, numExamples, trueJoint.getDifference2()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointInclExcl[idx][j].getIntersection();}, numExamples, trueJoint.getIntersection()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointInclExcl[idx][j].get1();}, numExamples, trueJoint.get1()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointInclExcl[idx][j].get2();}, numExamples, trueJoint.get2()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointInclExcl[idx][j].getUnion();}, numExamples, trueJoint.getUnion()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointInclExcl[idx][j].getJaccard();}, numExamples, trueJoint.getJaccard()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointInclExcl[idx][j].getCosine();}, numExamples, trueJoint.getCosine()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointInclExcl[idx][j].getInclusionCoefficient1();}, numExamples, trueJoint.getInclusionCoefficient1()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointInclExcl[idx][j].getInclusionCoefficient2();}, numExamples, trueJoint.getInclusionCoefficient2()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointInclExcl[idx][j].getAlpha();}, numExamples, trueJoint.getAlpha()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJointInclExcl[idx][j].getBeta();}, numExamples, trueJoint.getBeta()) << ";";
        
        f << calculateMean([&](uint64_t idx){return estJaccardColl[idx][j].first;}, numExamples) << ";";
        f << calculateMean([&](uint64_t idx){return estJaccardColl[idx][j].second;}, numExamples) << ";";
        f << calculateMSE([&](uint64_t idx){return estJaccardColl[idx][j].first;}, numExamples, trueJoint.getJaccard()) << ";";
        f << calculateMSE([&](uint64_t idx){return estJaccardColl[idx][j].second;}, numExamples, trueJoint.getJaccard()) << ";";

        f << endl;
        
    }

    f.close();

}

int main(int argc, char* argv[]) {

    mt19937_64 dataSeedRng(0xf5c1f864cefbf048);
    mt19937_64 configSeedRng(0x79b5d54d2f1489da);

    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint8_t>>(16384, 2., 30, 62, configSeedRng()), false);
    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint8_t>>(16384, 1.2, 30, std::numeric_limits<uint8_t>::max() - 1, configSeedRng()), false);
    test(dataSeedRng(), SetSketchConfig1<RegistersWithLowerBound<uint16_t>>(16384, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()), false);

    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint8_t>>(16384, 2., 30, 62, configSeedRng()), false);
    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint8_t>>(16384, 1.2, 30, std::numeric_limits<uint8_t>::max() - 1, configSeedRng()), false);
    test(dataSeedRng(), SetSketchConfig2<RegistersWithLowerBound<uint16_t>>(16384, 1.001, 30, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()), false);

    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(16384, 2., 62, configSeedRng()), true);
    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint8_t>>(16384, 1.2, std::numeric_limits<uint8_t>::max() - 1, configSeedRng()), true);
    test(dataSeedRng(), GeneralizedHyperLogLogConfig<RegistersWithLowerBound<uint16_t>>(16384, 1.001, std::numeric_limits<uint16_t>::max() - 1, configSeedRng()), true);

}