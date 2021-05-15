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

#ifndef _SKETCH_HPP_
#define _SKETCH_HPP_

#include <cstdint>
#include <vector>
#include <map>
#include <iostream>
#include <cassert>
#include <iterator>
#include <limits>
#include <cmath>

#include <boost/math/tools/roots.hpp>
#include <boost/math/tools/minima.hpp>

#include "bitstream_random.hpp"

// returns a histogram from the given data
// returns a vector of value/frequency pairs
template<typename T>
static std::vector<std::pair<typename T::value_type, uint64_t>> createHistogram(const T& data) {
    std::unordered_map<typename T::value_type, uint64_t> tmp;
    for(auto d : data) tmp[d] += 1;
    return std::vector<std::pair<typename T::value_type, uint64_t>>(tmp.begin(), tmp.end());
}

// x / (e^x-1)
static double xDivExpm1(double x) {
    return (x != 0.) ? x / std::expm1(x) : 1.;
}

// (e^x-1) / x
static double expm1DivX(double x) {
    return (x != 0.) ? std::expm1(x) / x : 1.;
}

// log(1+x) / x
static double log1pDivX(double x) {
    return (x != 0.) ? std::log1p(x) / x : 1.;
}

class JointEstimationResult {
    double difference1Cardinality;
    double difference2Cardinality;
    double intersectionCardinality;

    JointEstimationResult(double difference1Cardinality, double difference2Cardinality, double intersectionCardinality) :
        difference1Cardinality(difference1Cardinality),
        difference2Cardinality(difference2Cardinality),
        intersectionCardinality(intersectionCardinality) {
            assert(difference1Cardinality >= 0);
            assert(difference2Cardinality >= 0);
            assert(intersectionCardinality >= 0);
    }

public:

    double getDifference1() const {return difference1Cardinality;}

    double getDifference2() const {return difference2Cardinality;}

    double get1() const {return difference1Cardinality + intersectionCardinality;}

    double get2() const {return difference2Cardinality + intersectionCardinality;}

    double getUnion() const {return difference1Cardinality + difference2Cardinality + intersectionCardinality;}

    double getIntersection() const {return intersectionCardinality;}

    double getJaccard() const {return (intersectionCardinality > 0)?intersectionCardinality / (difference1Cardinality + difference2Cardinality + intersectionCardinality) : 0.;}

    double getCosine() const {return (intersectionCardinality > 0)?intersectionCardinality / std::sqrt((difference1Cardinality + intersectionCardinality)*(difference2Cardinality + intersectionCardinality)) : 0.;}

    double getInclusionCoefficient1() const {return (intersectionCardinality > 0)?intersectionCardinality / (difference1Cardinality + intersectionCardinality):0;}

    double getInclusionCoefficient2() const {return (intersectionCardinality > 0)?intersectionCardinality / (difference2Cardinality + intersectionCardinality):0;}

    double getAlpha() const {return (difference1Cardinality > 0)?difference1Cardinality / (difference1Cardinality + difference2Cardinality + intersectionCardinality):0.;}

    double getBeta() const {return (difference2Cardinality > 0)?difference2Cardinality / (difference1Cardinality + difference2Cardinality + intersectionCardinality):0.;}

    static JointEstimationResult createEmpty() {
        return JointEstimationResult(0., 0., 0.);
    }

    static JointEstimationResult createFromCardinalitiesAndJaccardSimilarity(double card1, double card2, double jaccardSimilarity) {
        assert(card1 >= 0);
        assert(card2 >= 0);
        if(card1 == 0. && card2 == 0.) return createEmpty();

        const double jaccardSimilarityUpperBound = std::min(card1 / card2, card2 / card1);
        const double trimmedJaccardSimilarity = std::max(0., std::min(jaccardSimilarityUpperBound, jaccardSimilarity));
        const double y = 1. / (1 + trimmedJaccardSimilarity);
        const double difference1Cardinality = std::max(0., card1 - card2*trimmedJaccardSimilarity) * y;
        const double difference2Cardinality = std::max(0., card2 - card1*trimmedJaccardSimilarity) * y;
        const double intersectionCardinality = (static_cast<double>(card1) + static_cast<double>(card2)) * trimmedJaccardSimilarity * y;
        return JointEstimationResult(difference1Cardinality, difference2Cardinality, intersectionCardinality);
    }

    static JointEstimationResult createFromCardinalitiesAndUnion(double card1, double card2, double cardUnion) {
        assert(card1 >= 0);
        assert(card2 >= 0);
        assert(cardUnion >= 0);
        return createFromCardinalitiesAndJaccardSimilarity(card1, card2, (card1 + card2 - cardUnion) / cardUnion);
    }

    static JointEstimationResult createFromCardinalitiesAndAlphaBetaDeprecated(double card1, double card2, double alpha, double beta) {
        double z = 1 - alpha - beta;
        if (z >= 0)  {
            double cardUnion = (card1 + card2) / (1 + z);
            return JointEstimationResult(cardUnion * alpha, cardUnion * beta, cardUnion * z);
        } else {
            // intersection estimate = 0, assuming sketches represent disjoint sets
            return JointEstimationResult(card1, card2, 0);
        }
    }

    static JointEstimationResult createFromTrueCardinalities(uint64_t card1Minus2, uint64_t card2Minus1, uint64_t cardIntersection) {
        return JointEstimationResult(card1Minus2, card2Minus1, cardIntersection);
    }
};

class SetSketchEstimator {
    const uint64_t q;
    const double a;
    const double base;
    const double baseInverse;
    const uint64_t numRegisters;
    const double logBaseDivBaseMinus1;
    const double factor;
    std::vector<double> tauValues;
    std::vector<double> sigmaValues;
    std::vector<double> baseInversePowers;
    const double logBase;
    const double logBaseInverse;
    const double basem1p2;
    const double basem1p3;
    const bool useCardinalityRangeCorrection;
    const bool useJointRangeCorrection;
    const bool useJointFallBack;

    double p_inv1(double y) const {
        return std::min(1., -std::expm1(-logBase*y)*(base / (base - 1.)));
    }

    // b1mxm1 is b^(1-x)-1
    // onembmx is 1 - b^(-x)
    // bmx = b^(-x)
    double uPrime(double b1mxm1, double onembmx, double bmx, uint64_t l) const {
        assert(l >= 1);
        assert(l <= q + 1);

        double onembmxBaseInversePowersM1 = onembmx * baseInversePowers[l-1];
        double onembmxBaseInversePowers = onembmx * baseInversePowers[l];

        double numerator = baseInversePowers[l] * bmx * basem1p3 * (b1mxm1 - onembmxBaseInversePowersM1);

        double hh1 = b1mxm1 + onembmxBaseInversePowers;
        double hh2 = b1mxm1 + onembmxBaseInversePowersM1;
        double hh3 = b1mxm1 + onembmxBaseInversePowersM1 * base;

        double logbArgument = b1mxm1 * onembmxBaseInversePowers * basem1p2 / (hh2 * hh2);
        double logB = std::log1p(logbArgument) * logBaseInverse;

        double denominator = logB * hh1 * hh2 * hh3;

        double result = numerator / denominator;
        assert(!std::isnan(result));

        return result;

    }

    double solveJointMLEquation (
        const uint64_t numEqual,
        const std::vector<std::pair<uint64_t, uint64_t>>& delta1Larger2,
        const std::vector<std::pair<uint64_t, uint64_t>>& delta2Larger1) const {
        if (delta1Larger2.empty()) {
            return 0;
        } else {
            std::pair<double, double> intAlphaPrime = boost::math::tools::bisect(
                [&](double alpha){
                    if (alpha <= 0) return -std::numeric_limits<double>::infinity();
                    if (alpha >= 1) return std::numeric_limits<double>::infinity();
                    double sum1 = 0;
                    double b1malpham1 = std::expm1(logBase*(1. - alpha));
                    double onembmalpha= -std::expm1(-logBase*alpha);
                    double bmalpha = 1. - onembmalpha;

                    for(auto x : delta1Larger2) {
                        sum1 += x.second * uPrime(b1malpham1, onembmalpha, bmalpha, x.first);
                    }
                    if (numEqual == 0) {
                        return -sum1;
                    } else if (delta2Larger1.empty()) {
                        return numEqual / (1. - alpha) - sum1;
                    } else {
                        if (sum1 <= 0) return std::numeric_limits<double>::infinity();
                        double beta = 1. - alpha - numEqual / sum1;
                        if (beta <= 0) return std::numeric_limits<double>::infinity();

                        double b1mbetam1 = std::expm1(logBase*(1. - beta));
                        double onembmbeta= -std::expm1(-logBase*beta);
                        double bmbeta = 1. - onembmbeta;

                        double sum2 = 0;
                        for(auto x : delta2Larger1) {
                            sum2 += x.second * uPrime(b1mbetam1, onembmbeta, bmbeta, x.first);
                        }
                        return sum2 - sum1;
                    }
                },
                0.,
                1.,
                [](double a, double b){return std::abs(a-b) <= 0;});

            return (intAlphaPrime.first + intAlphaPrime.second) * 0.5;
        }
    }

    double sigma(double x) const {
        assert(base > 1);
        assert(x >= 0);
        assert(x <= 1);

        if (x == 0.) return 0;
        if (x == 1.) return std::numeric_limits<double>::infinity();

        double sum = 0;
        double xbk = x;
        double bkm1 = 1;
        double oldSum;
        while(true) {
            oldSum = sum;
            xbk = std::pow(xbk, base);
            sum += xbk * bkm1;
            if (oldSum == sum) break;
            bkm1 *= base;
        }
        return x + (base - 1) * sum;
    }

    double tau(double x) const {
        if (x == 0. || x == 1.) return 0.;

        double sum = 0;
        double xbmk = x;
        double bmk = baseInverse;
        double oldSum;

        while(true) {
            oldSum = sum;
            sum += (xbmk - 1) * bmk;
            if (oldSum == sum) break;
            xbmk = std::pow(xbmk, baseInverse);
            bmk *= baseInverse;
        }
        return (1 - x) + (base - 1) * sum;
    }

public:

    SetSketchEstimator(uint64_t q, double a, double base, uint64_t numRegisters, bool useCardinalityRangeCorrection, bool useJointRangeCorrection, bool useJointFallBack):
        q(q),
        a(a),
        base(base),
        baseInverse(1./base),
        numRegisters(numRegisters),
        logBaseDivBaseMinus1(log1pDivX(base-1)),
        factor(numRegisters/(base*logBaseDivBaseMinus1*a)),
        tauValues(numRegisters+1),
        sigmaValues(numRegisters+1),
        baseInversePowers(q+2),
        logBase(std::log(base)),
        logBaseInverse(1./logBase),
        basem1p2((base - 1) * (base - 1)),
        basem1p3(basem1p2 * (base - 1)),
        useCardinalityRangeCorrection(useCardinalityRangeCorrection),
        useJointRangeCorrection(useJointRangeCorrection),
        useJointFallBack(useJointFallBack) {

        for(uint64_t i = 0; i <= q + 1; ++i) {
            baseInversePowers[i] = std::pow(base, -static_cast<double>(i));
        }
        for(uint64_t i = 1; i <= q + 1; ++i) {
            assert(baseInversePowers[i] <= baseInversePowers[i-1]);
        }
        for(uint32_t i = 0; i <= numRegisters; ++i) {
            tauValues[i] = numRegisters * baseInversePowers[q]*tau(static_cast<double>(numRegisters - i) / static_cast<double>(numRegisters));
            sigmaValues[i] = numRegisters * sigma(static_cast<double>(i) / static_cast<double>(numRegisters));
        }
   }

    template<typename S>
    double estimateCardinalitySimple(const S& state) const {
        double sum = 0;
        uint64_t numRegisterValuesEqualMin = 0;
        uint64_t numRegisterValuesEqualMax = 0;
        for(auto r : state) {
            if (useCardinalityRangeCorrection && r == 0) {
                numRegisterValuesEqualMin += 1;
            } else if (useCardinalityRangeCorrection && r > q) {
                numRegisterValuesEqualMax += 1;
            } else {
                sum += baseInversePowers[r];
            }
        }
        if (useCardinalityRangeCorrection) {
            sum += sigmaValues[numRegisterValuesEqualMin];
            sum += tauValues[numRegisterValuesEqualMax];
        }
        return factor / sum;
    }

    template<typename S>
    double estimateCardinalityML(const S& state) const {
        auto registerHistogram = createHistogram(state);
        double z = 0;
        uint64_t count0 = 0;
        for(auto valueFrequencyPair : registerHistogram) {

            if(!useCardinalityRangeCorrection || valueFrequencyPair.first <= q) {
                if(useCardinalityRangeCorrection && valueFrequencyPair.first == 0) {
                    count0 = valueFrequencyPair.second;
                    if (count0 == numRegisters) return 0;
                }
                z += valueFrequencyPair.second * baseInversePowers[valueFrequencyPair.first];
            } else {
                if (valueFrequencyPair.second == numRegisters) return std::numeric_limits<double>::infinity();
            }
        }
        z *= a;

        const double cardinalityLowerBound = 0;
        const double cardinalityUpperBound = (numRegisters - count0) / z;

        auto maxIterations = std::numeric_limits<boost::uintmax_t>::max();
        auto result = boost::math::tools::toms748_solve(
            [&](double n){
                double na = n * a;
                double nabasem1 = na * (base - 1);
                double y = 0;
                for(auto valueFrequencyPair : registerHistogram) {
                    if (useCardinalityRangeCorrection && valueFrequencyPair.first == q + 1) {
                        y += valueFrequencyPair.second * xDivExpm1(na * baseInversePowers[q]);
                    } else if (!useCardinalityRangeCorrection || valueFrequencyPair.first > 0) {
                        y += valueFrequencyPair.second * xDivExpm1(nabasem1 * baseInversePowers[valueFrequencyPair.first]);
                    }
                }
                return y - n * z;
            },
            cardinalityLowerBound,
            cardinalityUpperBound,
            [](double a, double b) {return std::abs(b - a) <= 1e-9 * std::max(a, b);}, maxIterations);
        return (result.first + result.second) * 0.5;
    }

    template<typename S>
    std::vector<double> estimateCardinalities(const S& state) const {
        return {estimateCardinalitySimple(state), estimateCardinalityML(state)};
    }

    static std::vector<std::string> getCardinalityEstimateLabels() {return {"simple","ml"};}

    template<typename S>
    JointEstimationResult estimateJointInclExcl(const S& state1, const S& state2) const {
        double card1 = estimateCardinalitySimple(state1);
        double card2 = estimateCardinalitySimple(state2);
        return estimateJointInclExclWithKnownSetCardinalities(card1, card2, state1, state2);
    }

    template<typename S>
    JointEstimationResult estimateJointInclExclWithKnownSetCardinalities(double card1, double card2, const S& state1, const S& state2) const {
        std::vector<uint64_t> unionRegisterValues(numRegisters);
        auto it1 = state1.begin();
        auto it2 = state2.begin();
        for(uint32_t idx = 0; idx < numRegisters; ++idx) {
            unionRegisterValues[idx]= std::max(*it1, *it2);
            ++it1;
            ++it2;
        }
        double unionCardinality = estimateCardinalitySimple(unionRegisterValues);

        return JointEstimationResult::createFromCardinalitiesAndUnion(card1, card2, unionCardinality);
    }

    template<typename S>
    JointEstimationResult estimateJointSimpleDeprecated(const S& state1, const S& state2) const {

        uint64_t numRegisters1Less2 = 0;
        uint64_t numRegisters1Greater2 = 0;
        bool hasEqualRegistersWithExtremeValues = false;
        for(uint32_t idx = 0; idx < numRegisters; ++idx) {
            auto val1 = state1.getRegisterValue(idx);
            auto val2 = state2.getRegisterValue(idx);
            assert(val1 <= q + 1);
            assert(val2 <= q + 1);

            if (val1 < val2) {
                numRegisters1Less2 += 1;
            }
            else if (val1 > val2) {
                numRegisters1Greater2 += 1;
            } else {
                if (val1 == 0 || val1 == q + 1) hasEqualRegistersWithExtremeValues = true;
            }
        }

        if(useJointFallBack && hasEqualRegistersWithExtremeValues) {
            // fall back to inclusion-exclusion principle
            return estimateJointInclExcl(state1, state2);
        }
        double alphaPrime = static_cast<double>(numRegisters1Greater2) / static_cast<double>(numRegisters);
        double betaPrime = static_cast<double>(numRegisters1Less2) / static_cast<double>(numRegisters);
        double alpha = p_inv1(alphaPrime);
        double beta = p_inv1(betaPrime);
        double card1 = estimateCardinalitySimple(state1);
        double card2 = estimateCardinalitySimple(state2);
        return JointEstimationResult::createFromCardinalitiesAndAlphaBetaDeprecated(card1, card2, alpha, beta);
    }

    template<typename S>
    JointEstimationResult estimateJointMLDeprecated(const S& state1, const S& state2) const {

        uint64_t numEqual = 0;
        std::vector<uint64_t> deltas1Larger2Raw;
        std::vector<uint64_t> deltas2Larger1Raw;
        bool registersWithExtremeValues = false;
        for(uint32_t idx = 0; idx < numRegisters; ++idx) {
            uint64_t val1 = state1.getRegisterValue(idx);
            uint64_t val2 = state2.getRegisterValue(idx);
            if (val1 == 0 || val1 == q + 1 || val2 == 0 || val2 == q + 1) registersWithExtremeValues = true;
            if (val1 > val2) {
                deltas1Larger2Raw.push_back(val1 - val2);
            } else if (val2 > val1) {
                deltas2Larger1Raw.push_back(val2 - val1);
            } else {
                numEqual += 1;
            }
        }

        if(useJointFallBack && registersWithExtremeValues) {
            // fall back to inclusion-exclusion principle
            return estimateJointInclExcl(state1, state2);
        }

        auto delta1Larger2 = createHistogram(deltas1Larger2Raw);
        auto delta2Larger1 = createHistogram(deltas2Larger1Raw);

        double alphaPrime = solveJointMLEquation(numEqual, delta1Larger2, delta2Larger1);
        double betaPrime = solveJointMLEquation(numEqual, delta2Larger1, delta1Larger2);
        double alpha = p_inv1(alphaPrime);
        double beta = p_inv1(betaPrime);
        double card1 = estimateCardinalitySimple(state1);
        double card2 = estimateCardinalitySimple(state2);
        return JointEstimationResult::createFromCardinalitiesAndAlphaBetaDeprecated(card1, card2, alpha, beta);
    }

    JointEstimationResult estimateJointNew(
        const uint64_t numRegisters1Less2,
        const uint64_t numRegisters1Greater2,
        const double card1,
        const double card2) const {

        const uint64_t numEqual = numRegisters - numRegisters1Less2 - numRegisters1Greater2;

        if (card1 == 0. && card2 == 0.) return JointEstimationResult::createEmpty();

        const double z = (1. - baseInverse) / (card1 + card2);

        const double domainMin = 0;
        const double domainMax = (card1 >= card2) ? card2/card1 : card1/card2;

        std::pair<double, double> r = boost::math::tools::brent_find_minima(
            [&](double j) {

                double log1px1 = (numEqual > 0 || numRegisters1Greater2 > 0) ? logBaseInverse * std::log1p((card2 * j - card1) * z) : 0.;
                double log1px2 = (numEqual > 0 || numRegisters1Less2 > 0) ? logBaseInverse * std::log1p((card1 * j - card2) * z) : 0.;

                double ret = 0;
                if (numEqual > 0) ret += numEqual * std::log1p(log1px1 + log1px2);
                if (numRegisters1Greater2 > 0) ret += numRegisters1Greater2 * std::log(-log1px1);
                if (numRegisters1Less2 > 0) ret += numRegisters1Less2 * std::log(-log1px2);

                if(std::isnan(ret)) { // can happen due to numerical inaccuracies at the domain boundary,
                                      // set to infinity to ensure that minimization can proceed in this case
                    return std::numeric_limits<double>::infinity();
                }
                return -ret;
            },
            domainMin,
            domainMax,
            std::numeric_limits<double>::digits); // use large enough value to get best possible result,
                                                  // see boost documentation of brent_find_minima
        double estimatedJaccard = r.first;
        return JointEstimationResult::createFromCardinalitiesAndJaccardSimilarity(card1, card2, estimatedJaccard);
    }

    // calculates sum_{k=0}^infinity 1 - e^{-x*b^{-k}}
   /* double zz(double x) const {
        assert(x >= 0);
        double sum = 0.;
        double pow = 1.;
        while(true) {
            double oldSum = sum;
            sum -= std::expm1(-x*pow);
            if (sum == oldSum) return sum;
            pow *= baseInverse;
        }
    }*/

    double mu(double x, double y) const {
        assert(x >= 0);
        assert(y >= 0);
        if(x == y) return 0.;
        bool swapped = (x > y);
        if (swapped) std::swap(x,y);
        assert(x < y);
        double result = 0.;

        uint64_t kMid;
        if (x > 0) {
            const double z = std::log(std::log(x / y) / (x-y)) * logBaseInverse;
            kMid = static_cast<uint64_t>(std::floor(std::max(0., z)));
        } else {
            kMid = 0;
        }

        const double powBaseMid = std::pow(baseInverse, kMid);
        {
            double powBase = powBaseMid;
            while(true) {
                double oldResult = result;
                result -= std::exp(-x*powBase)*std::expm1((x-y)*powBase);
                if (oldResult == result) break;
                powBase *= baseInverse;
            }
        }
        {
            double powBase = powBaseMid;
            uint64_t k = kMid;
            while(k != 0) {
                powBase *= base;
                double oldResult = result;
                result -= std::exp(-x*powBase)*std::expm1((x-y)*powBase);
                if (oldResult == result) break;
                k -= 1;
            }
        }
        return (swapped)?(-result):result;
    }

    JointEstimationResult estimateJointNewCorrected(
        const uint64_t numRegisters1Less2NotBothZero,
        const uint64_t numRegisters1Greater2NotBothZero,
        const uint64_t numBothZero,
        const double card1,
        const double card2) const {

        assert(numBothZero > 0);

        if (card1 == 0. && card2 == 0.) return JointEstimationResult::createEmpty();

        const double domainMin = 0;
        const double domainMax = (card1 >= card2) ? card2/card1 : card1/card2;

        assert(numRegisters1Less2NotBothZero + numRegisters1Greater2NotBothZero + numBothZero <= numRegisters);
        const uint64_t numEqualNotBothZero = numRegisters - numRegisters1Less2NotBothZero - numRegisters1Greater2NotBothZero - numBothZero;

        std::pair<double, double> r = boost::math::tools::brent_find_minima(
            [&](double j) {
                const double c = a * (card1 + card2) / (1 + j);

                double pRegisters1Greater2NotBothZero =
                    (numRegisters1Greater2NotBothZero > 0 || numEqualNotBothZero > 0)?
                        mu(a*(std::max(0., card1 - card2*j)/(base*(1+j)) + card2), c):0.;

                double pRegisters1Less2NotBothZero =
                    (numRegisters1Less2NotBothZero > 0 || numEqualNotBothZero > 0)?
                        mu(a*(std::max(0.,card2 - card1*j)/(base*(1+j)) + card1), c):0.;

                // double pRegisters1Greater2NotBothZero =
                //     (numRegisters1Greater2NotBothZero > 0 || numEqualNotBothZero > 0)?
                //         (zz(c) - zz(a*((card1 - card2*j)/(base*(1+j)) + card2))):0.;

                // double pRegisters1Less2NotBothZero =
                //     (numRegisters1Less2NotBothZero > 0 || numEqualNotBothZero > 0)?
                //         (zz(c) - zz(a*((card2 - card1*j)/(base*(1+j)) + card1))):0.;


                double ret = -c * numBothZero;
                if (numRegisters1Less2NotBothZero > 0) ret += numRegisters1Less2NotBothZero * std::log(pRegisters1Less2NotBothZero);
                if (numRegisters1Greater2NotBothZero > 0) ret += numRegisters1Greater2NotBothZero * std::log(pRegisters1Greater2NotBothZero);
                if (numEqualNotBothZero > 0) ret += numEqualNotBothZero * std::log1p(-pRegisters1Greater2NotBothZero - pRegisters1Less2NotBothZero - std::exp(-c));

                if(std::isnan(ret)) { // can happen due to numerical inaccuracies at the domain boundary,
                                      // set to infinity to ensure that minimization can proceed in this case
                    return std::numeric_limits<double>::infinity();
                }
                return -ret;
            },
            domainMin,
            domainMax,
            std::numeric_limits<double>::digits); // use large enough value to get best possible result,
                                                  // see boost documentation of brent_find_minima
        double estimatedJaccard = r.first;
        return JointEstimationResult::createFromCardinalitiesAndJaccardSimilarity(card1, card2, estimatedJaccard);
    }

    template<typename S>
    JointEstimationResult estimateJointNew(const S& state1, const S& state2) const {
        const double card1 = estimateCardinalitySimple(state1);
        const double card2 = estimateCardinalitySimple(state2);
        return estimateJointNewWithKnownSetCardinalities(card1, card2, state1, state2);
    }

    template<typename S>
    JointEstimationResult estimateJointNewWithKnownSetCardinalities(double card1, double card2, const S& state1, const S& state2) const {
        uint64_t numRegisters1Less2 = 0;
        uint64_t numRegisters1Greater2 = 0;
        uint64_t numRegistersBothMin = 0;
        uint64_t numRegistersBothMax = 0;
        auto it1 = state1.begin();
        auto it2 = state2.begin();
        for(uint32_t idx = 0; idx < numRegisters; ++idx) {
            auto val1 = *it1;
            auto val2 = *it2;
            ++it1;
            ++it2;
            assert(val1 <= q + 1);
            assert(val2 <= q + 1);

            if (val1 < val2) {
                numRegisters1Less2 += 1;
            } else if (val1 > val2) {
                numRegisters1Greater2 += 1;
            } else if (val1 == 0) {
                numRegistersBothMin += 1;
            } else if (val1 == q + 1) {
                numRegistersBothMax += 1;
            }
        }

        if (numRegistersBothMin > 0 || numRegistersBothMax > 0) {
            if(useJointRangeCorrection && numRegistersBothMax == 0) {
                return estimateJointNewCorrected(numRegisters1Less2, numRegisters1Greater2, numRegistersBothMin, card1, card2);
            } else if (useJointFallBack) {
                return estimateJointInclExclWithKnownSetCardinalities(card1, card2, state1, state2);
            }
        }

        return estimateJointNew(numRegisters1Less2, numRegisters1Greater2, card1, card2);
    }

    template<typename S>
    std::vector<JointEstimationResult> estimateJointQuantities(uint64_t card1, uint64_t card2, const S& state1, const S& state2) const {
        return {
            estimateJointNew(state1, state2),
            estimateJointNewWithKnownSetCardinalities(card1, card2, state1, state2),
            estimateJointInclExcl(state1, state2),
            estimateJointInclExclWithKnownSetCardinalities(card1, card2, state1, state2)};
    }

    static std::vector<std::string> getJointEstimateLabels() {
        return {
            "new",
            "newKnownCard",
            "inclExcl",
            "inclExclKnownCard"};
    }

    static constexpr bool canEstimateJaccardSimilarityUsingEqualRegisters() {
        return true;
    }

    template<typename S>
    std::pair<double,double> estimateJaccardSimilarityUsingEqualRegisters(const S& state1, const S& state2) const {
        uint64_t numEqual = 0;
        for(uint32_t idx = 0; idx < numRegisters; ++idx) {
            uint64_t val1 = state1.getRegisterValue(idx);
            uint64_t val2 = state2.getRegisterValue(idx);
            if (val1 == val2) {
                numEqual += 1;
            }
        }

        double g = static_cast<double>(numEqual)/static_cast<double>(numRegisters);
        double h = (g + 1.)*0.5;

        double lowerBoundEstimate = std::max(0., 2.*(expm1DivX(logBase * h)*log1pDivX(base-1)*h)-1.);
        double upperBoundEstimate = expm1DivX(logBase * g)*log1pDivX(base-1)*g;

        return std::make_pair(lowerBoundEstimate, upperBoundEstimate);
    }
};

class Mapping {
    std::vector<double> baseInversePowers;
    const uint64_t searchIncrement;
    static constexpr double skipProbability = 0.5;
public:
    Mapping(double base, uint64_t q) :
        baseInversePowers(q+1),
        searchIncrement(static_cast<uint64_t>(std::floor(std::max(1., -std::log(skipProbability) / std::log(base))))) {
        for(uint64_t i = 0; i <= q; ++i) baseInversePowers[i] = std::pow(base, -static_cast<double>(i));
        assert(searchIncrement > 0);
    }

    uint64_t map(uint64_t kLow, double x) const {

        uint64_t q = baseInversePowers.size()-1;

        if(kLow > q || !(x <= baseInversePowers[kLow])) return kLow;
        uint64_t k = kLow + searchIncrement;
        while(k <= q && x <= baseInversePowers[k]) k += searchIncrement;
        uint64_t kSearchIntervalMin = k - searchIncrement;
        uint64_t kSearchIntervalMax;
        if (k <= q) {
            kSearchIntervalMax = k;
        } else {
            if (x <= baseInversePowers[q]) return q + 1;
            kSearchIntervalMax = q;
        }

        while(kSearchIntervalMin + 1 != kSearchIntervalMax) {
            uint64_t kSearchIntervalMid = (kSearchIntervalMin + kSearchIntervalMax) >> 1;
            if (x <= baseInversePowers[kSearchIntervalMid]) {
                kSearchIntervalMin = kSearchIntervalMid;
             } else {
                kSearchIntervalMax = kSearchIntervalMid;
             }
        }
        return kSearchIntervalMax;
    }

    bool isRelevant(uint64_t kLow, double x) const {
        uint64_t q = baseInversePowers.size() - 1;
        return x <= 0 || (kLow <= q && x <= baseInversePowers[kLow]);
    }
};

template<typename R>
class RegistersWithLowerBound{
    std::vector<R> registerValues;
    R registerValueLowerBound;
    uint64_t numberOfUpdatesToLowerBoundUpdate;

    static constexpr double numberOfUpdatesRatio = 1.;

    void resetNumberOfUpdatesToLowerBoundUpdate() {
        numberOfUpdatesToLowerBoundUpdate = static_cast<uint64_t>(std::floor(numberOfUpdatesRatio * registerValues.size()));
    }
public:

    typedef R value_type;

    template<typename C>
    RegistersWithLowerBound(const C& config) :
        registerValues(config.getNumRegisters(), 0),
        registerValueLowerBound(0) {
        resetNumberOfUpdatesToLowerBoundUpdate();
    }

    // all register values are equal to or greater than the returned lower bound
    R getRegisterValueLowerBound() const {
        return registerValueLowerBound;
    }

    void update(uint32_t registerIdx, R newRegisterValue) {
        if (newRegisterValue > registerValueLowerBound) {
            uint32_t oldRegisterValue = registerValues[registerIdx];
            if (newRegisterValue > oldRegisterValue) {
                registerValues[registerIdx] = newRegisterValue;
                numberOfUpdatesToLowerBoundUpdate -= 1;
                if (numberOfUpdatesToLowerBoundUpdate == 0) getRegisterValueMinimum();
            }
        }
    }

    R getRegisterValue(uint32_t registerIdx) const {
        return registerValues[registerIdx];
    }

    typename std::vector<R>::const_iterator begin() const {
        return registerValues.begin();
    }

    typename std::vector<R>::const_iterator end() const {
        return registerValues.end();
    }

    template<typename S>
    void merge(const S& otherState) {
        registerValueLowerBound = std::numeric_limits<R>::max();
        for(uint32_t registerIdx = 0; registerIdx < registerValues.size(); registerIdx+=1) {
            registerValues[registerIdx] = std::max(registerValues[registerIdx], otherState.getRegisterValue(registerIdx));
            registerValueLowerBound = std::min(registerValueLowerBound, registerValues[registerIdx]);
        }
        resetNumberOfUpdatesToLowerBoundUpdate();
    }

    // after calling this function, getRegisterValueLowerBound() will return the minimum of all register values
    R getRegisterValueMinimum() {
        registerValueLowerBound = *std::min_element(std::begin(registerValues), std::end(registerValues));
        resetNumberOfUpdatesToLowerBoundUpdate();
        return registerValueLowerBound;
    }

    static std::string getDescription() {
        return "registers with lower bound";
    }

    bool operator==(const RegistersWithLowerBound& other) const {
        return registerValues == other.registerValues;
    }
};

template<typename R>
class Registers {
    std::vector<R> registerValues;
public:

    typedef R value_type;

    template<typename C>
    Registers(const C& config) : registerValues(config.getNumRegisters(), 0){}

    constexpr R getRegisterValueLowerBound() const {
        return 0;
    }

    void update(uint32_t registerIdx, R newRegisterValue) {
        uint32_t oldRegisterValue = registerValues[registerIdx];
        if (newRegisterValue > oldRegisterValue) {
            registerValues[registerIdx] = newRegisterValue;
        }
    }

    R getRegisterValue(uint32_t registerIdx) const {
        return registerValues[registerIdx];
    }

    typename std::vector<R>::const_iterator begin() const {
        return registerValues.begin();
    }

    typename std::vector<R>::const_iterator end() const {
        return registerValues.end();
    }

    template<typename S>
    void merge(const S& otherState) {
        for(uint32_t registerIdx = 0; registerIdx < registerValues.size(); registerIdx+=1) {
            registerValues[registerIdx] = std::max(registerValues[registerIdx], otherState.getRegisterValue(registerIdx));
        }
    }

    static std::string getDescription() {
        return "registers";
    }

    R getRegisterValueMinimum() const {
        return *std::min_element(std::begin(registerValues), std::end(registerValues));
    }

    bool operator==(const Registers& other) const {
        return registerValues == other.registerValues;
    }
};

template<typename C> class HyperLogLog;
template<typename C> class GeneralizedHyperLogLog;
template<typename C> class SetSketch1;
template<typename C> class SetSketch2;
template<typename C> class HyperMinHash;
class MinHash;

template<typename S>
class GeneralizedHyperLogLogConfig {
public:
    typedef WyrandBitStream bit_stream_type;
    typedef typename S::value_type value_type;
    typedef S state_type;
    typedef GeneralizedHyperLogLog<GeneralizedHyperLogLogConfig> sketch_type;
private:
    const uint32_t numRegisters;
    const double base;
    const uint64_t q;
    const double a;
    const SetSketchEstimator estimator;
    const Mapping mapping;
    const double inverseLogBase;
public:

    GeneralizedHyperLogLogConfig(uint32_t numRegisters, double base, uint64_t q) :
            numRegisters(numRegisters),
            base(base),
            q(q),
            a(1./numRegisters),
            estimator(q, 1. / numRegisters, base, numRegisters, true, false, false), // set 2nd last or last parameter to true to activate small range correction or fall back to inclusion exclusion principle respectively
            mapping(base, q),
            inverseLogBase(1./std::log(base)) {

        assert(numRegisters > 0);
        assert(base > 1);
        assert(std::numeric_limits<value_type>::max() > q);
    }

    bool operator==(const GeneralizedHyperLogLogConfig& config) const {
        if (numRegisters != config.numRegisters) return false;
        if (base != config.base) return false;
        if (q != config.q) return false;
        return true;
    }

    uint32_t getNumRegisters() const {return numRegisters;}

    bit_stream_type getBitStream(uint64_t x) const {return bit_stream_type(x);}

    double getBase() const {return base;}

    uint64_t getQ() const {return q;}

    double getA() const {return a;}

    double getInverseLogBase() const {return inverseLogBase;}

    const SetSketchEstimator& getEstimator() const {return estimator;}

    GeneralizedHyperLogLog<GeneralizedHyperLogLogConfig<S>> create() const {return GeneralizedHyperLogLog(*this);}

    std::string getName() const {return "GeneralizedHyperLogLog";}

    const Mapping& getMapping() const {return mapping;}
};

template<typename S>
class HyperLogLogConfig {
public:
    typedef typename S::value_type value_type;
    typedef S state_type;
    typedef HyperLogLog<HyperLogLogConfig> sketch_type;
private:
    const uint32_t numRegisters;
    const uint64_t p;
    const uint64_t q;
    const double a;
    const SetSketchEstimator estimator;
public:

    HyperLogLogConfig(uint64_t p, uint64_t q) :
            numRegisters(UINT32_C(1) << p),
            p(p),
            q(q),
            a(1./numRegisters),
            estimator(q, a, 2., numRegisters, true, false, false) { // set 2nd last or last parameter to true to activate small range correction or fall back to inclusion exclusion principle respectively

        assert(p + q <= 64);
        assert(numRegisters > 0);
        assert(std::numeric_limits<value_type>::max() > q);
    }

    bool operator==(const HyperLogLogConfig& config) const {
        if (p != config.p) return false;
        if (q != config.q) return false;
        return true;
    }

    uint32_t getNumRegisters() const {return numRegisters;}

    double getBase() const {return 2.;}

    uint64_t getP() const {return p;}

    uint64_t getQ() const {return q;}

    double getA() const {return a;}

    const SetSketchEstimator& getEstimator() const {return estimator;}

    HyperLogLog<HyperLogLogConfig<S>> create() const {return HyperLogLog(*this);}

    std::string getName() const {return "HyperLogLog";}
};

template<typename C>
class BaseSketch {
protected:
    const C& config;
    typename C::state_type state;
public:

    BaseSketch(const C& config) : config(config), state(config) {}

    void merge(const BaseSketch<C>& other) {
        assert(config == other.getConfig());
        state.merge(other.state);
    }

    const C& getConfig() const {
        return config;
    }

    const typename C::state_type& getState() const {
        return state;
    }
};

template<typename C>
class GeneralizedHyperLogLog : public BaseSketch<C> {
public:

    GeneralizedHyperLogLog(const C& config) : BaseSketch<C>{config} {}

    void add(uint64_t d) {
        const auto& c = this->config;
        auto bitstream = c.getBitStream(d);
        double x = getUniformDouble(bitstream);
        auto kLow = this->state.getRegisterValueLowerBound();
        auto k = c.getMapping().map(kLow, x);
        if (k == kLow) return;
        uint32_t registerIdx = getUniformLemire(c.getNumRegisters(), bitstream);
        this->state.update(registerIdx, k);
    }

    /*void add(uint64_t d) {
        const auto& c = this->config;
        auto bitstream = c.getBitStream(d);
        double x = ziggurat::getExponential(bitstream);
        typename C::value_type k = static_cast<typename C::value_type>(1 + std::min(x * c.getInverseLogBase(), static_cast<double>(c.getQ())));
        auto kLow = this->state.getRegisterValueLowerBound();
        if (k <= kLow) return;
        uint32_t registerIdx = getUniformLemire(c.getNumRegisters(), bitstream);
        this->state.update(registerIdx, k);
    }*/
};

template<typename C>
class HyperLogLog : public BaseSketch<C> {
public:

    HyperLogLog(const C& config) : BaseSketch<C>{config} {}

    void add(uint64_t d) {

        // performance improvement
        if (((UINT64_C(0xFFFFFFFFFFFFFFFF) << this->state.getRegisterValueLowerBound()) | d) == UINT64_C(0xFFFFFFFFFFFFFFFF)) {
            const auto& c = this->config;
            uint32_t registerIdx = static_cast<uint32_t>(d >> (64 - c.getP()));
            uint8_t k = 1;
            while(k <= c.getQ() && (d & 1)) {
                k += 1;
                d >>= 1;
            }
            this->state.update(registerIdx, k);
        }
    }
};

static constexpr double bulkAddFirstAttemptSuccessProbabilityDefault = 0.95;

template<typename S>
class SetSketchConfig1 {
public:
    typedef WyrandBitStream bit_stream_type;
    typedef typename S::value_type value_type;
    typedef S state_type;
    typedef SetSketch1<SetSketchConfig1> sketch_type;
private:
    const uint32_t numRegisters;
    double base;
    const uint64_t q;
    const double a;
    const SetSketchEstimator estimator;
    std::vector<double> factors;
    const Mapping mapping;
    const double limitFactor;
    const double bulkAddFirstAttemptSuccessProbability;
public:

    SetSketchConfig1(uint32_t numRegisters, double base, double a, uint64_t q, double bulkAddFirstAttemptSuccessProbability = bulkAddFirstAttemptSuccessProbabilityDefault) :
            numRegisters(numRegisters),
            base(base),
            q(q),
            a(a),
            estimator(q, a, base, numRegisters, false, false, false),
            factors(numRegisters),
            mapping(base, q),
            limitFactor(-std::log1p(-std::pow(bulkAddFirstAttemptSuccessProbability, 1. / numRegisters)) / a),
            bulkAddFirstAttemptSuccessProbability(bulkAddFirstAttemptSuccessProbability)
    {

        assert(numRegisters > 0);
        assert(base > 1);
        assert(std::numeric_limits<value_type>::max() > q);

        for(uint32_t i = 0; i < numRegisters; ++i) factors[i] = 1./(a * (numRegisters - i));
    }

    bool operator==(const SetSketchConfig1& config) const {
        if (numRegisters != config.numRegisters) return false;
        if (base != config.base) return false;
        if (a != config.a) return false;
        if (q != config.q) return false;
        return true;
    }

    uint32_t getNumRegisters() const {return numRegisters;}

    bit_stream_type getBitStream(uint64_t x) const {return bit_stream_type(x);}

    double getBase() const {return base;}

    uint64_t getQ() const {return q;}

    double getA() const {return a;}

    double getFactor(uint32_t i) const {return factors[i];}

    const SetSketchEstimator& getEstimator() const {return estimator;}

    SetSketch1<SetSketchConfig1<S>> create() const {return SetSketch1(*this);}

    std::string getName() const {return "SetSketch1";}

    const Mapping& getMapping() const {return mapping;}

    double getLimitFactor() const {return limitFactor;}

    double getBulkAddFirstAttemptSuccessProbability() const {return bulkAddFirstAttemptSuccessProbability;}
};

struct NoPointStopCondition {
    constexpr bool operator()(double x) const {return false;}
};

class LimitPointStopCondition {
    double limit;
public:
    LimitPointStopCondition(double limit) : limit(limit) {}
    bool operator()(double x) const {return x > limit;}
};

template<typename C>
class SetSketch1 : public BaseSketch<C> {

    template<typename P>
    void addHelper(uint64_t d, const P& pointStopCondition) {
        auto bitstream = this->config.getBitStream(d);
        PermutationStream::reset(this->config.getNumRegisters());
        const auto& c = this->config;
        const uint32_t m = c.getNumRegisters();
        double x = 0;
        for(uint32_t i = 0; i < m; ++i) {
            x += ziggurat::getExponential(bitstream) * c.getFactor(i);
            if (pointStopCondition(x)) break;
            uint64_t kLow = this->state.getRegisterValueLowerBound();
            uint64_t k = c.getMapping().map(kLow, x);
            if (k == kLow) return;
            uint32_t registerIdx = PermutationStream::next(bitstream);
            this->state.update(registerIdx, k);
        }
    }

public:

    SetSketch1(const C& config) : BaseSketch<C>{config} {}

    SetSketch1(const SetSketch1& sketch) : BaseSketch<C>(sketch) {}

    void add(uint64_t d) {
        addHelper(d, NoPointStopCondition());
    }

    template<typename T>
    void addAll(const T& data, uint64_t* numAttempts = nullptr) {
        const double pointLimitIncrement = this->config.getLimitFactor() / std::size(data);
        double pointLimit = pointLimitIncrement;
        uint64_t numAttemptsCounter = 1;
        while(true) {
            for(uint64_t d : data) {
                addHelper(d, LimitPointStopCondition(pointLimit));
            }

            uint64_t kLow = this->state.getRegisterValueMinimum();
            uint64_t k = this->config.getMapping().map(kLow, pointLimit);
            if(k == kLow) {
                if (numAttempts != nullptr) (*numAttempts) = numAttemptsCounter;
                return; // success
            }

            // otherwise retry
            pointLimit += pointLimitIncrement;
            numAttemptsCounter += 1;
        }
    }
};

template<typename S>
class SetSketchConfig2 {
public:
    typedef WyrandBitStream bit_stream_type;
    typedef typename S::value_type value_type;
    typedef S state_type;
    typedef SetSketch2<SetSketchConfig2> sketch_type;
private:
    const uint32_t numRegisters;
    const double base;
    const uint64_t q;
    const double a;
    const SetSketchEstimator estimator;
    std::vector<double> gammaTimesAInv;
    std::vector<TruncatedExponentialDistribution> truncatedExponentialDistributions;
    const double aInv;
    const Mapping mapping;
    const double limitFactor;
    const double bulkAddFirstAttemptSuccessProbability;
public:

    SetSketchConfig2(uint32_t numRegisters, double base, double a, uint64_t q, double bulkAddFirstAttemptSuccessProbability = bulkAddFirstAttemptSuccessProbabilityDefault) :
            numRegisters(numRegisters),
            base(base),
            q(q),
            a(a),
            estimator(q, a, base, numRegisters, false, false, false),
            gammaTimesAInv(numRegisters),
            truncatedExponentialDistributions(numRegisters - 1),
            aInv(1. / a),
            mapping(base, q),
            limitFactor(-std::log1p(-std::pow(bulkAddFirstAttemptSuccessProbability, 1. / numRegisters)) / a), // this formula is only an approximation, as it ignores the correlation of register values of SetSketch2 for small cardinalities
            bulkAddFirstAttemptSuccessProbability(bulkAddFirstAttemptSuccessProbability)
    {

        assert(numRegisters > 0);
        assert(base > 1);
        assert(std::numeric_limits<value_type>::max() > q);

        for(uint32_t i = 0; i < numRegisters - 1; ++i) truncatedExponentialDistributions[i] = TruncatedExponentialDistribution(std::log1p(1./static_cast<double>(numRegisters - i - 1)));
        for(uint32_t i = 0; i < numRegisters; ++i) gammaTimesAInv[i] =
            std::log1p(static_cast<double>(i)/static_cast<double>(numRegisters - i)) * aInv;
    }

    bool operator==(const SetSketchConfig2& config) const {
        if (numRegisters != config.numRegisters) return false;
        if (base != config.base) return false;
        if (a != config.a) return false;
        if (q != config.q) return false;
        return true;
    }

    uint32_t getNumRegisters() const {return numRegisters; }

    bit_stream_type getBitStream(uint64_t x) const {return bit_stream_type(x);}

    double getBase() const {return base;}

    uint64_t getQ() const {return q;}

    double getA() const {return a;}

    double getAInv() const {return aInv;}

    double getGammaTimesAInv(uint32_t i) const {
        assert(i < numRegisters);
        return gammaTimesAInv[i];
    }

    const TruncatedExponentialDistribution& getTruncatedExponentialDistribution(uint32_t i) const {
        assert(i < numRegisters - 1);
        return truncatedExponentialDistributions[i];
    }

    const SetSketchEstimator& getEstimator() const {return estimator;}

    SetSketch2<SetSketchConfig2<S>> create() const {return SetSketch2(*this);}

    std::string getName() const {return "SetSketch2";}

    const Mapping& getMapping() const {return mapping;}

    double getLimitFactor() const {return limitFactor;}

    double getBulkAddFirstAttemptSuccessProbability() const {return bulkAddFirstAttemptSuccessProbability;}
};

template<typename C>
class SetSketch2 : public BaseSketch<C> {

    template<typename P>
    void addHelper(uint64_t d, const P& pointStopCondition) {
        auto bitstream = this->config.getBitStream(d);
        PermutationStream::reset(this->config.getNumRegisters());
        const auto& c = this->config;
        const uint32_t m = c.getNumRegisters();

        for(uint32_t i = 0; i < m; ++i) {
            double gammaTimesAInvI = c.getGammaTimesAInv(i);
            if (pointStopCondition(gammaTimesAInvI)) break;
            uint64_t kLow = this->state.getRegisterValueLowerBound();
            if (!c.getMapping().isRelevant(kLow, gammaTimesAInvI)) break;
            double x;
            if (i < m - 1) {
                x = gammaTimesAInvI + (c.getGammaTimesAInv(i+1) - gammaTimesAInvI) * c.getTruncatedExponentialDistribution(i)(bitstream);
            } else {
                x = gammaTimesAInvI + c.getAInv() * ziggurat::getExponential(bitstream);
            }
            if (pointStopCondition(x)) break;
            uint64_t k = c.getMapping().map(kLow, x);
            if (k == kLow) return;
            uint32_t registerIdx = PermutationStream::next(bitstream);
            this->state.update(registerIdx, k);
        }
    }

public:

    SetSketch2(const C& config) : BaseSketch<C>{config} {}

    SetSketch2(const SetSketch2& sketch) : BaseSketch<C>(sketch) {}

    void add(uint64_t d) {
        addHelper(d, NoPointStopCondition());
    }

    template<typename T>
    void addAll(const T& data, uint64_t* numAttempts = nullptr) {
        const double pointLimitIncrement = this->config.getLimitFactor() / std::size(data);
        double pointLimit = pointLimitIncrement;
        uint64_t numAttemptsCounter = 1;
        while(true) {
            for(uint64_t d : data) {
                addHelper(d, LimitPointStopCondition(pointLimit));
            }

            uint64_t kLow = this->state.getRegisterValueMinimum();
            uint64_t k = this->config.getMapping().map(kLow, pointLimit);
            if(k == kLow) {
                if (numAttempts != nullptr) (*numAttempts) = numAttemptsCounter;
                return; // success
            }

            // otherwise retry
            pointLimit += pointLimitIncrement;
            numAttemptsCounter += 1;
        }
    }
};

class MinHashEstimator {
    const uint64_t numRegisters;
    //static constexpr double log2 = std::log(2.);
    static constexpr double log2Times64 = std::log(2.)*64;
public:

    MinHashEstimator(uint64_t numRegisters):
        numRegisters(numRegisters) {}

    double estimateCardinality(const std::vector<uint64_t>& state) const {
        double sum = 0;
        for(uint64_t r : state) {
            sum += log2Times64 - std::log(~r);
        }
        return numRegisters / sum;
    }

    /*double estimateCardinality(const std::vector<uint64_t>& state) const {
        double prod = 1;
        int64_t exponent = state.size() << 6;
        for(uint64_t i = 0; i < state.size(); ++i) {
            prod *= ~state[i];
            if ((i & UINT64_C(7)) == UINT64_C(7)) {
                int t;
                prod = std::frexp(prod, &t);
                exponent -= t;
            }
        }
        return static_cast<double>(numRegisters) / (exponent * log2 - std::log(prod));
    }*/

    static uint64_t numCardinalityEstimates() {return 2;}

    std::vector<double> estimateCardinalities(const std::vector<uint64_t>& state) const {
        double estimate = estimateCardinality(state);
        return {estimate, estimate};
    }

    static std::vector<std::string> getCardinalityEstimateLabels() {return {"simple", "ml"};}

    JointEstimationResult estimateJointInclExclWithKnownSetCardinalities(double card1, double card2, const std::vector<uint64_t>& state1, const std::vector<uint64_t>& state2) const {
        std::vector<uint64_t> unionState(numRegisters);
        for(uint32_t idx = 0; idx < numRegisters; ++idx) {
             unionState[idx]= std::min(state1[idx], state2[idx]);
        }
        double unionCardinality = estimateCardinality(unionState);
        return JointEstimationResult::createFromCardinalitiesAndUnion(card1, card2, unionCardinality);
    }

    JointEstimationResult estimateJointInclExcl(const std::vector<uint64_t>& state1, const std::vector<uint64_t>& state2) const {
        double card1 = estimateCardinality(state1);
        double card2 = estimateCardinality(state2);
        return estimateJointInclExclWithKnownSetCardinalities(card1, card2, state1, state2);
    }

    JointEstimationResult estimateJointSimpleDeprecated(const std::vector<uint64_t>& state1, const std::vector<uint64_t>& state2) const {

        uint64_t numRegisters1Less2 = 0;
        uint64_t numRegisters1Greater2 = 0;
        for(uint32_t idx = 0; idx < numRegisters; ++idx) {
            auto val1 = state1[idx];
            auto val2 = state2[idx];
            if (val1 < val2) {
                numRegisters1Less2 += 1;
            }
            else if (val1 > val2) {
                numRegisters1Greater2 += 1;
            }
        }

        double alpha = static_cast<double>(numRegisters1Less2) / static_cast<double>(numRegisters);
        double beta = static_cast<double>(numRegisters1Greater2) / static_cast<double>(numRegisters);
        double card1 = estimateCardinality(state1);
        double card2 = estimateCardinality(state2);

        return JointEstimationResult::createFromCardinalitiesAndAlphaBetaDeprecated(card1, card2, alpha, beta);
    }

    JointEstimationResult estimateJointOriginalWithKnownSetCardinalities(double card1, double card2, const std::vector<uint64_t>& state1, const std::vector<uint64_t>& state2) const {
        uint64_t numRegistersEqual = 0;
        for(uint32_t idx = 0; idx < numRegisters; ++idx) {
            auto val1 = state1[idx];
            auto val2 = state2[idx];
            if (val1 == val2) numRegistersEqual += 1;
        }
        double estimatedJaccard = static_cast<double>(numRegistersEqual) / static_cast<double>(numRegisters);
        return JointEstimationResult::createFromCardinalitiesAndJaccardSimilarity(card1, card2, estimatedJaccard);
    }

    JointEstimationResult estimateJointOriginal(const std::vector<uint64_t>& state1, const std::vector<uint64_t>& state2) const {
        const double card1 = estimateCardinality(state1);
        const double card2 = estimateCardinality(state2);
        return estimateJointOriginalWithKnownSetCardinalities(card1, card2, state1, state2);
    }

    JointEstimationResult estimateJointNew(const std::vector<uint64_t>& state1, const std::vector<uint64_t>& state2) const {
        const double card1 = estimateCardinality(state1);
        const double card2 = estimateCardinality(state2);
        return estimateJointNewWithKnownSetCardinalities(card1, card2, state1, state2);
    }

    JointEstimationResult estimateJointNewWithKnownSetCardinalities(double card1, double card2, const std::vector<uint64_t>& state1, const std::vector<uint64_t>& state2) const {
        uint64_t numRegisters1Less2 = 0;
        uint64_t numRegisters1Greater2 = 0;
        for(uint32_t idx = 0; idx < numRegisters; ++idx) {
            auto val1 = state1[idx];
            auto val2 = state2[idx];
            if (val1 < val2) {
                numRegisters1Less2 += 1;
            }
            else if (val1 > val2) {
                numRegisters1Greater2 += 1;
            }
        }

        const double d0 = numRegisters - numRegisters1Less2 - numRegisters1Greater2;
        const double dp = numRegisters1Less2;
        const double dm = numRegisters1Greater2;
        const double na2 = card1 * card1;
        const double nb2 = card2 * card2;
        const double xa = na2 * (d0 + dm);
        const double xb = nb2 * (d0 + dp);

        const double estimatedJaccard = std::min(1., std::max(0.,
            (xa + xb - std::sqrt(std::pow(xa - xb, 2) + 4. * dp * dm * na2 * nb2))
            /
            (2. * numRegisters * card1 * card2)));

        return JointEstimationResult::createFromCardinalitiesAndJaccardSimilarity(card1, card2, estimatedJaccard);
    }

    std::vector<JointEstimationResult> estimateJointQuantities(uint64_t card1, uint64_t card2, const std::vector<uint64_t>& state1, const std::vector<uint64_t>& state2) const {
        return {
            estimateJointNew(state1, state2),
            estimateJointNewWithKnownSetCardinalities(card1, card2, state1, state2),
            estimateJointInclExcl(state1, state2),
            estimateJointInclExclWithKnownSetCardinalities(card1, card2, state1, state2),
            estimateJointSimpleDeprecated(state1, state2),
            estimateJointOriginal(state1, state2),
            estimateJointOriginalWithKnownSetCardinalities(card1, card2, state1, state2)};
    }

    static std::vector<std::string> getJointEstimateLabels() {
        return {
            "new",
            "newKnownCard",
            "inclExcl",
            "inclExclKnownCard",
            "simpleDeprecated",
            "original",
            "originalKnownCard"};
    }

    static constexpr bool canEstimateJaccardSimilarityUsingEqualRegisters() {
        return true;
    }

    std::pair<double,double> estimateJaccardSimilarityUsingEqualRegisters(const std::vector<uint64_t>& state1, const std::vector<uint64_t>& state2) const {
        uint64_t numEqual = 0;
        for(uint32_t idx = 0; idx < numRegisters; ++idx) {
            uint64_t val1 = state1[idx];
            uint64_t val2 = state2[idx];
            if (val1 == val2) {
                numEqual += 1;
            }
        }

        double estimate = static_cast<double>(numEqual) / static_cast<double>(numRegisters);
        return std::make_pair(estimate, estimate); // lower and upper bound estimates are equal for MinHash
    }
};

class MinHashConfig {
public:
    typedef uint64_t value_type;
    typedef MinHash sketch_type;
    typedef WyrandBitStream bit_stream_type;
private:
    const uint64_t numRegisters;
    const MinHashEstimator estimator;
public:

    MinHashConfig(const uint32_t numRegisters) :
            numRegisters(numRegisters),
            estimator(numRegisters) {
        assert(numRegisters > 0);
    }

    bool operator==(const MinHashConfig& config) const {
        if (numRegisters != config.numRegisters) return false;
        return true;
    }

    uint32_t getNumRegisters() const {return numRegisters;}

    bit_stream_type getBitStream(uint64_t x) const {return bit_stream_type(x);}

    const MinHashEstimator& getEstimator() const {return estimator;}

    MinHash create() const;

    std::string getName() const {return "MinHash";}
};


class MinHash {
    std::vector<uint64_t> state;
    const MinHashConfig& config;
public:
    MinHash(const MinHashConfig& config) : state(config.getNumRegisters(), std::numeric_limits<uint64_t>::max()), config(config) {}

    void add(uint64_t d) {
        auto bitstream = this->config.getBitStream(d);
        for(uint64_t& s : state) {
            s = std::min(s, bitstream(64));
        }
    }

    template<typename T>
    void addAll(const T& data) {
        for(uint64_t d : data) {
            add(d);
        }
    }

    void merge(const MinHash& other) {
        for(uint32_t registerIdx = 0; registerIdx < state.size(); registerIdx+=1) {
            state[registerIdx] = std::min(state[registerIdx], other.state[registerIdx]);
        }
    }

    const MinHashConfig& getConfig() const {
        return config;
    }

    const std::vector<uint64_t>& getState() const {
        return state;
    }
};


MinHash MinHashConfig::create() const {
    return MinHash(*this);
}

static double calculateEffectiveBaseForHyperMinHash(uint64_t subbucketsize) {
    return std::pow(2., 1./(UINT64_C(1) << subbucketsize));
}

class HyperMinHashEstimator {
    const uint32_t numRegisters;
    const uint64_t bucketbits;
    const uint64_t bucketsize;
    const uint64_t subbucketsize;
    const double alpha;
    const uint64_t subbucketmask;
    const double limit;
    const uint64_t effectiveQ;
    const double effectiveBase;
    const double effectiveA;
    const SetSketchEstimator jointEstimator;
    const SetSketchEstimator cardinalityEstimator;

    static double getAlpha(uint32_t numRegisters) {
        if (numRegisters == 16) {
            return 0.673;
        } else if (numRegisters == 32) {
            return 0.697;
        } else if (numRegisters == 64) {
            return 0.709;
        } else {
            return 0.7213 / (1 + 1.079 / numRegisters);
        }
    }

    // based on https://github.com/yunwilliamyu/hyperminhash/blob/3a91004f76a9d5f69624e2926f41a82d8b0c2015/hyperminhash.py#L276
    template<typename S>
    double hll_estimator(const S& state) const {
        double sum = 0.;
        uint64_t V = 0;
        for(uint64_t r : state) {
            uint64_t rHLL = r >> subbucketsize;
            sum += 1. / static_cast<double>(UINT64_C(1) << rHLL);
            if (rHLL == 0) V += 1;
        }
        double res = alpha * numRegisters * numRegisters / sum;

        double res2;
        if (res <= (5. / 2.) * numRegisters) {
            if (V != 0) {
                res2 = numRegisters * std::log(numRegisters / static_cast<double>(V));  // linear counting
            } else {
                res2 = res;
            }
        } else if (res <= (1. / 30.) * (UINT64_C(1) << 32)) {
            res2 = res;
        } else {
            res2 = -(UINT64_C(1) << 32) * std::log(1. - res / (UINT64_C(1) << 32));
        }
        return res2;
    }

    template<typename S>
    std::vector<uint64_t> transformToGeneralizedHyperLogLogState(const S& state) const {
        std::vector<uint64_t> result;
        for(uint64_t s : state) {
            uint64_t rHLL = s >> subbucketsize;
            if (rHLL == 0) {
                result.push_back(0);
            } else {
                assert(s >= subbucketmask +1 );
                result.push_back(std::min(s - subbucketmask, effectiveQ + 1));
            }
        }
        return result;
    }

    template<typename S>
    std::vector<uint64_t> transformToHyperLogLogState(const S& state) const {
        std::vector<uint64_t> result;
        for(uint64_t s : state) {
            uint64_t rHLL = s >> subbucketsize;
            result.push_back(rHLL);
        }
        return result;
    }

public:

    HyperMinHashEstimator(uint64_t bucketbits, uint64_t bucketsize, uint64_t subbucketsize) :
        numRegisters(UINT32_C(1) << bucketbits),
        bucketbits(bucketbits),
        bucketsize(bucketsize),
        subbucketsize(subbucketsize),
        alpha(getAlpha(numRegisters)),
        subbucketmask((UINT64_C(1) << subbucketsize) - 1),
        limit(std::pow(2, bucketbits + 10)),
        effectiveQ(((UINT64_C(1) << bucketsize) - 1) * (UINT64_C(1) << subbucketsize)),
        effectiveBase(calculateEffectiveBaseForHyperMinHash(subbucketsize)),
        effectiveA(1./numRegisters),
        jointEstimator(effectiveQ, effectiveA, effectiveBase, numRegisters, true, false, false), // set 2nd last to true to activate fall back to inclusion exclusion principle
        cardinalityEstimator(((UINT64_C(1) << bucketsize) - 1), effectiveA, 2., numRegisters, true, false, false) {

        assert(subbucketsize <= 63);
    }

    // based on https://github.com/yunwilliamyu/hyperminhash/blob/3a91004f76a9d5f69624e2926f41a82d8b0c2015/hyperminhash.py#L154
    template<typename S>
    double estimateCardinalityOriginal(const S& state) const {
        double hll_count = hll_estimator(state);
        if (hll_count < limit && bucketsize > 0) {
            return hll_count;
        } else {
            double sum = 0;
            for(uint64_t r : state) {
                uint64_t rHLL = r >> subbucketsize;
                uint64_t x = r & subbucketmask;
                sum += (1. + static_cast<double>(x) / static_cast<double>(UINT64_C(1) << subbucketsize)) / (UINT64_C(1) << rHLL);
            }
            return numRegisters * numRegisters / sum;
        }
    }

    // https://github.com/yunwilliamyu/hyperminhash/blob/3a91004f76a9d5f69624e2926f41a82d8b0c2015/hyperminhash.py#L340
    double collision_estimate_hll_divided(double x_size, double y_size) const {
        double cp = 0;
        double n = x_size;
        double m = y_size;
        double num_hll_buckets = std::pow(2., bucketsize);

        for(uint64_t i_ = 0; i_ < num_hll_buckets; ++i_) {
            uint64_t i = i_ + 1;
            double b1;
            double b2;
            if (i != num_hll_buckets) {
                b1 = 1. / std::pow(2., i);
                b2 = 1. / std::pow(2., i - 1);
            } else {
                b1 = 0;
                b2 = 1. / std::pow(2., i - 1);
            }
            b1 = b1 / numRegisters;
            b2 = b2 / numRegisters;
            double pr_x = std::pow(1 - b1, n) - std::pow(1 - b2, n);
            double pr_y = std::pow(1 - b1, m) - std::pow(1 - b2, m);
            cp += pr_x * pr_y;
        }
        return (cp * numRegisters) / std::pow(2., subbucketsize);
    }

    // https://github.com/yunwilliamyu/hyperminhash/blob/3a91004f76a9d5f69624e2926f41a82d8b0c2015/hyperminhash.py#L365
    double collision_estimate_final(double x_size, double y_size) const {

        double n = std::max(x_size, y_size);
        double m = std::min(x_size, y_size);

        assert(n <= numRegisters * std::pow(2., std::pow(2., bucketsize) + subbucketsize - 10.));
        if (n > numRegisters * 32.) {
            double ratio = n / m;
            double ratio_factor = 4. * ratio / std::pow(1. + ratio, 2);
            return (0.169919487159739093975315012348630288992889 * numRegisters * ratio_factor) / std::pow(2., subbucketsize);
        } else {
            return collision_estimate_hll_divided(x_size, y_size);
        }
    }

    // compare https://github.com/axiomhq/hyperminhash/blob/8f66e1a155488b0434f845f1e4783e61dc0571c9/hyperminhash.go#L103
    template<typename S>
    JointEstimationResult estimateJointOriginalWithKnownSetCardinalities(double card1, double card2, const S& state1, const S& state2) const {
        // self_nonzeros = np.logical_or(self.hll != 0, self.bbit != 0)
        // matches_with_zeros = np.logical_and(self.hll == other.hll, self.bbit == other.bbit)
        // matches = np.logical_and(self_nonzeros, matches_with_zeros)
        // match_num = sum(matches)
        uint64_t match_num = 0;
        uint64_t union_filled_buckets = 0;
        for(uint64_t registerIdx = 0; registerIdx < numRegisters; ++registerIdx) {
            if(state1.getRegisterValue(registerIdx) != 0 && state1.getRegisterValue(registerIdx) == state2.getRegisterValue(registerIdx)) match_num += 1;
            if(state1.getRegisterValue(registerIdx) != 0 || state2.getRegisterValue(registerIdx) != 0) union_filled_buckets += 1;
        }

        double collisions = collision_estimate_final(card1, card2);

        double intersect_size = match_num - collisions;
        double estimatedJaccard;

        // this condition was added compared to the reference implementation to increase robustness
        // the problem that this condition may not be satisfied has also been found in this Go implementation
        // https://github.com/axiomhq/hyperminhash/blob/8f66e1a155488b0434f845f1e4783e61dc0571c9/hyperminhash.go#L122
        if (intersect_size >= 0) {
            if(union_filled_buckets == 0 || intersect_size <= 0) {
                estimatedJaccard = 0.;
            } else {
                estimatedJaccard = intersect_size / union_filled_buckets;
            }
        } else {
           estimatedJaccard = 0.;
        }

        return JointEstimationResult::createFromCardinalitiesAndJaccardSimilarity(card1, card2, estimatedJaccard);
    }

    template<typename S>
    JointEstimationResult estimateJointOriginal(const S& state1, const S& state2) const {
        // use better new cardinality estimator instead of original one, to have a fair comparison of joint estimator
        double card1 = estimateCardinalityNew(state1);
        double card2 = estimateCardinalityNew(state2);
        return estimateJointOriginalWithKnownSetCardinalities(card1, card2, state1, state2);
    }

    template<typename S>
    double estimateCardinalityNew(const S& state) const {
        return cardinalityEstimator.estimateCardinalitySimple(transformToHyperLogLogState(state));
    }

    template<typename S>
    std::vector<double> estimateCardinalities(const S& state) const {
        return {estimateCardinalityOriginal(state), estimateCardinalityNew(state)};
    }

    static std::vector<std::string> getCardinalityEstimateLabels() {return {"original", "new"};}

    template<typename S>
    JointEstimationResult estimateJointNew(const S& state1, const S& state2) const {
        double card1 = estimateCardinalityNew(state1);
        double card2 = estimateCardinalityNew(state2);
        auto result = estimateJointNewWithKnownSetCardinalities(card1, card2, state1, state2);
        return result;
    }

    template<typename S>
    JointEstimationResult estimateJointNewWithKnownSetCardinalities(double card1, double card2, const S& state1, const S& state2) const {
        auto result = jointEstimator.estimateJointNewWithKnownSetCardinalities(card1, card2, transformToGeneralizedHyperLogLogState(state1), transformToGeneralizedHyperLogLogState(state2));
        return result;
    }

    template<typename S>
    JointEstimationResult estimateJointInclExcl(const S& state1, const S& state2) const {
        return cardinalityEstimator.estimateJointInclExcl(transformToHyperLogLogState(state1), transformToHyperLogLogState(state2));
    }

    template<typename S>
    std::vector<JointEstimationResult> estimateJointQuantities(uint64_t card1, uint64_t card2, const S& state1, const S& state2) const {
        return {
            estimateJointOriginal(state1, state2),
            estimateJointOriginalWithKnownSetCardinalities(card1, card2, state1, state2),
            estimateJointNew(state1, state2),
            estimateJointNewWithKnownSetCardinalities(card1, card2, state1, state2),
            estimateJointInclExcl(state1, state2)};
    }

    static std::vector<std::string> getJointEstimateLabels() {
        return {"original", "originalKnownCard", "new", "newKnownCard", "inclExcl"};
    }

    static constexpr bool canEstimateJaccardSimilarityUsingEqualRegisters() {
        return false;
    }
};

template<typename S>
class HyperMinHashConfig {
public:
    typedef typename S::value_type value_type;
    typedef S state_type;
    typedef HyperMinHash<HyperMinHashConfig<S>> sketch_type;
    typedef WyrandBitStream bit_stream_type;
private:
    const uint32_t numRegisters;
    const uint64_t bucketbits;
    const uint64_t bucketsize;
    const uint64_t subbucketsize;
    const HyperMinHashEstimator estimator;
    const double effectiveBase;
public:

    // compare https://github.com/yunwilliamyu/hyperminhash/blob/3a91004f76a9d5f69624e2926f41a82d8b0c2015/hyperminhash.py#L60
    HyperMinHashConfig(uint64_t bucketbits, uint64_t bucketsize, uint64_t subbucketsize) :
            numRegisters(UINT32_C(1) << bucketbits),
            bucketbits(bucketbits),
            bucketsize(bucketsize),
            subbucketsize(subbucketsize),
            estimator(bucketbits, bucketsize, subbucketsize),
            effectiveBase(calculateEffectiveBaseForHyperMinHash(subbucketsize))
    {
        assert(bucketbits + subbucketsize <= 64);
        assert(subbucketsize >= 0);
        assert(bucketbits >= 0);
        assert(bucketsize >= 1);
        assert(bucketsize <= 6);

        // left-most 1-bit is in the range {1, 2, 3,... , 2^bucketsize}
        // including 0, (2^bucketsize+1) states are possible, therefore (bucketsize+1) bits are needed for representation
        uint64_t bitsRequired = bucketsize + 1 + subbucketsize;
        assert(std::numeric_limits<value_type>::digits >= bitsRequired); // ensure that value type has at least required bits
        assert(numRegisters > 0);
    }

    bool operator==(const HyperMinHashConfig& config) const {
        if (bucketbits != config.bucketbits) return false;
        if (bucketsize != config.bucketsize) return false;
        if (subbucketsize != config.subbucketsize) return false;
        return true;
    }

    uint32_t getNumRegisters() const {return numRegisters;}

    double getBase() const {return effectiveBase;}

    uint64_t getP() const {return bucketbits;}

    uint64_t getBucketBits() const {return bucketbits;}
    uint64_t getBucketSize() const {return bucketsize;}
    uint64_t getSubBucketSize() const {return subbucketsize;}
    bit_stream_type getBitStream(uint64_t x) const {return bit_stream_type(x);}

    const HyperMinHashEstimator& getEstimator() const {return estimator;}

    HyperMinHash<HyperMinHashConfig> create() const {
        return HyperMinHash(*this);
    }

    std::string getName() const {return "HyperMinHash";}
};

template<typename C>
class HyperMinHash : public BaseSketch<C> {
public:
    HyperMinHash(const C& config) : BaseSketch<C>{config} {}

    HyperMinHash(const HyperMinHash& sketch) : BaseSketch<C>(sketch) {}

    void add(uint64_t d) {
        auto bitstream = this->config.getBitStream(d);
        uint64_t y = bitstream(64);
        uint8_t val = 1;
        while(val < (UINT64_C(1) << this->config.getBucketSize()) && (y & 1)) {
            val += 1;
            y >>= 1;
        }

        uint64_t registerIdx = bitstream(this->config.getBucketBits());
        uint64_t aug = bitstream(this->config.getSubBucketSize());

        uint64_t updateValue = (val << this->config.getSubBucketSize()) | aug;
        this->state.update(registerIdx, updateValue);
    }
};

template<typename S>
S merge(const S& sketch1, const S& sketch2) {
    S result = sketch1;
    result.merge(sketch2);
    return result;
}

template<typename C>
void appendInfo(std::ostream& os, const C& config)
{
    std::ios_base::fmtflags f( os.flags() );
    os << "name=" << config.getName() << ";";
    os << "numRegisters=" << config.getNumRegisters() << ";";
    os << "q=" << config.getQ() << ";";
    os << "base=" << std::scientific << std::setprecision(std::numeric_limits< double >::max_digits10) << config.getBase() << ";";
    os << "a=" << std::scientific << std::setprecision(std::numeric_limits< double >::max_digits10) << config.getA() << ";";
    os << "registerStateType=" << C::state_type::getDescription() << ";";
    appendBulkAddFirstAttemptSuccessProbability(os, config);
    os.flags( f );
}

template<typename S>
void appendInfo(std::ostream& os, const HyperMinHashConfig<S>& config)
{
    std::ios_base::fmtflags f( os.flags() );
    os << "name=" << config.getName() << ";";
    os << "numRegisters=" << config.getNumRegisters() << ";";
    os << "base=" << std::scientific << std::setprecision(std::numeric_limits< double >::max_digits10) << config.getBase() << ";";
    os << "bucketBits=" << config.getBucketBits() << ";";
    os << "bucketSize=" << config.getBucketSize() << ";";
    os << "subBucketSize=" << config.getSubBucketSize() << ";";
    os << "registerStateType=" << S::getDescription() << ";";
    os.flags( f );
}

void appendInfo(std::ostream& os, const MinHashConfig& config)
{
    std::ios_base::fmtflags f( os.flags() );
    os << "name=" << config.getName() << ";";
    os << "base=" << std::scientific << std::setprecision(std::numeric_limits< double >::max_digits10) << 1. << ";";
    os << "numRegisters=" << config.getNumRegisters() << ";";
    os.flags( f );
}

template<typename C>
void appendBulkAddFirstAttemptSuccessProbability(std::ostream& os, const C& config) {
}

template<typename D>
void appendBulkAddFirstAttemptSuccessProbability(std::ostream& os, const SetSketchConfig1<D>& config) {
    os << "bulkAddFirstAttemptSuccessProbability=" << std::scientific << std::setprecision(std::numeric_limits< double >::max_digits10) << config.getBulkAddFirstAttemptSuccessProbability() << ";";
}

template<typename D>
void appendBulkAddFirstAttemptSuccessProbability(std::ostream& os, const SetSketchConfig2<D>& config) {
    os << "bulkAddFirstAttemptSuccessProbability=" << std::scientific << std::setprecision(std::numeric_limits< double >::max_digits10) << config.getBulkAddFirstAttemptSuccessProbability() << ";";
}

#endif // _SKETCH_HPP_