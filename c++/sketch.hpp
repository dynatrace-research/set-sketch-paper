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
#include <iostream>
#include <cassert>
#include <iterator>
#include <limits>
#include <cmath>

#include <boost/math/tools/roots.hpp>

#include "bitstream_random.hpp"

// returns a histogram from the given data
// returns a vector of value/frequency pairs
template<typename T>
static std::vector<std::pair<T, uint64_t>> createHistogram(const std::vector<T>& data) {
    std::unordered_map<T, uint64_t> tmp;
    for(auto d : data) tmp[d] += 1;
    return std::vector<std::pair<T, uint64_t>>(tmp.begin(), tmp.end());
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

    JointEstimationResult(double difference1Cardinality, double difference2Cardinality, double intersectionCardinality) :
        difference1Cardinality(difference1Cardinality),
        difference2Cardinality(difference2Cardinality),
        intersectionCardinality(intersectionCardinality) {
            assert(difference1Cardinality >= 0);
            assert(difference2Cardinality >= 0);
            assert(intersectionCardinality >= 0);
    }
};

class CardinalityEstimator {
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
        do {
            oldSum = sum;
            xbk = std::pow(xbk, base);
            sum += xbk * bkm1;
            if (oldSum == sum) break;
            bkm1 *= base;
        } while(true);
        return x + (base - 1) * sum;
    }

    double tau(double x) const {
        if (x == 0. || x == 1.) return 0.;

        double sum = 0;
        double xbmk = x;
        double bmk = baseInverse;
        double oldSum;

        do {
            oldSum = sum;
            xbmk = std::pow(xbmk, baseInverse);
            sum += (xbmk - 1) * bmk;
            if (oldSum == sum) break;
            bmk *= baseInverse;
        } while(oldSum != sum);
        return (1-x) + (base-1) * sum;
    }


    template<typename S>
    JointEstimationResult calculateJointResult(const S& state1, const S& state2, double alphaPrime, double betaPrime, bool useRangeCorrection) const {
        assert(alphaPrime >= 0);
        assert(alphaPrime <= 1);
        assert(betaPrime >= 0);
        assert(betaPrime <= 1);
        double alpha = p_inv1(alphaPrime);
        double beta = p_inv1(betaPrime);
        assert(alpha >= 0);
        assert(alpha <= 1);
        assert(beta >= 0);
        assert(beta <= 1);
        double z = 1 - alpha - beta;
        double card1 = estimateCardinalitySimple(state1.getHistogram(), useRangeCorrection);
        double card2 = estimateCardinalitySimple(state2.getHistogram(), useRangeCorrection);
        if (z >= 0)  {
            double unionCardinality = (card1 + card2) / (1 + z);
            // double unionCardinality = card1 * card2 * (2.-alpha-beta) / ((1 - alpha)*(1-alpha) * card1 + (1 - beta)*(1-beta) * card2);
            return JointEstimationResult(unionCardinality * alpha, unionCardinality * beta, unionCardinality * z);
        } else {
            // intersection estimate = 0, assuming sketches represent disjoint sets
            return JointEstimationResult(card1, card2, 0);
        }
    }

public:

    CardinalityEstimator(uint64_t q, double a, double base, uint64_t numRegisters):
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
        basem1p3(basem1p2 * (base - 1)) {

        for(uint64_t i = 0; i <= q + 1; ++i) {
            baseInversePowers[i] = std::pow(base, -static_cast<double>(i));
        }
        for(uint64_t i = 1; i <= q + 1; ++i) {
            assert(baseInversePowers[i] <= baseInversePowers[i-1]);
        }
        for(uint32_t i = 0; i <= numRegisters; ++i) {
            tauValues[i] = numRegisters * baseInversePowers[q+1]*tau(static_cast<double>(numRegisters - i) / static_cast<double>(numRegisters));
            sigmaValues[i] = numRegisters * sigma(static_cast<double>(i) / static_cast<double>(numRegisters));
        }
   }

    template<typename U>
    double estimateCardinalitySimple(const std::vector<std::pair<U,uint64_t>>& registerHistogram, bool useRangeCorrection) const {
        double sum = 0;
         for(auto valueFrequencyPair : registerHistogram) {
            if (useRangeCorrection && valueFrequencyPair.first == 0) {
                sum += sigmaValues[valueFrequencyPair.second];
            } else if (useRangeCorrection && valueFrequencyPair.first > q) {
                sum += tauValues[valueFrequencyPair.second];
            } else {
                sum += valueFrequencyPair.second * baseInversePowers[valueFrequencyPair.first];
            }
        }
        return factor / sum;
    }

    template<typename U>
    double estimateCardinalityML(const std::vector<std::pair<U,uint64_t>>& registerHistogram, bool useRangeCorrection) const {

        double z = 0;
        uint64_t count0 = 0;
        for(auto valueFrequencyPair : registerHistogram) {

            if(!useRangeCorrection || valueFrequencyPair.first <= q) {
                if(useRangeCorrection && valueFrequencyPair.first == 0) {
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
                    if (useRangeCorrection && valueFrequencyPair.first == q + 1) {
                        y += valueFrequencyPair.second * xDivExpm1(na * baseInversePowers[q]);
                    } else if (!useRangeCorrection || valueFrequencyPair.first > 0) {
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
    JointEstimationResult estimateJointInclExcl(const S& state1, const S& state2, bool useRangeCorrection) const {
        std::vector<uint64_t> unionRegisterValues(numRegisters);
        for(uint32_t idx = 0; idx < numRegisters; ++idx) {
             unionRegisterValues[idx]= std::max(state1.getRegisterValue(idx), state2.getRegisterValue(idx));
        }
        double unionCardinality = estimateCardinalitySimple(createHistogram(unionRegisterValues), useRangeCorrection);
        double card1 = estimateCardinalitySimple(state1.getHistogram(), useRangeCorrection);
        double card2 = estimateCardinalitySimple(state2.getHistogram(), useRangeCorrection);
        assert(std::isfinite(card1));
        assert(std::isfinite(card2));
        double card1Minus2 = unionCardinality - card2;
        double card2Minus1 = unionCardinality - card1;
        assert(card1Minus2 >= 0);
        assert(card2Minus1 >= 0);

        double intersection = card1 + card2 - unionCardinality;
        if (intersection >= 0) {
            return JointEstimationResult(card1Minus2, card2Minus1, intersection);
        } else {
            return JointEstimationResult(card1, card2, 0);
        }
    }

    template<typename S>
    JointEstimationResult estimateJointSimple(const S& state1, const S& state2, bool useRangeCorrection) const {

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

        if(useRangeCorrection && hasEqualRegistersWithExtremeValues) {
            // fall back to inclusion-exclusion principle
            return estimateJointInclExcl(state1, state2, useRangeCorrection);
        }
        double alphaPrime = static_cast<double>(numRegisters1Greater2) / static_cast<double>(numRegisters);
        double betaPrime = static_cast<double>(numRegisters1Less2) / static_cast<double>(numRegisters);
        return calculateJointResult(state1, state2, alphaPrime, betaPrime, useRangeCorrection);
    }

    template<typename S>
    JointEstimationResult estimateJointML(const S& state1, const S& state2, bool useRangeCorrection) const {

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

        if(useRangeCorrection && registersWithExtremeValues) {
            // fall back to inclusion-exclusion principle
            return estimateJointInclExcl(state1, state2, useRangeCorrection);
        }

        auto delta1Larger2 = createHistogram(deltas1Larger2Raw);
        auto delta2Larger1 = createHistogram(deltas2Larger1Raw);

        double alphaPrime = solveJointMLEquation(numEqual, delta1Larger2, delta2Larger1);
        double betaPrime = solveJointMLEquation(numEqual, delta2Larger1, delta1Larger2);

        return calculateJointResult(state1, state2, alphaPrime, betaPrime, useRangeCorrection);

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
        return kLow <= q && x <= baseInversePowers[kLow];
    }

    uint64_t getSearchIncrement() const {
        return searchIncrement;
    }
};


template<typename R>
class RegistersWithHistogram{
    std::vector<R> registerValues;
    std::vector<uint32_t> histogram;
    R minRegisterValue;
public:

    typedef R RegValueType;

    template<typename C>
    RegistersWithHistogram(const C& config) :
        registerValues(config.getNumRegisters(), 0),
        histogram(config.getQ() + 2, 0),
        minRegisterValue(0)
    {
        assert(static_cast<uint64_t>(config.getQ()) + 1 <= std::numeric_limits<R>::max());
        histogram[0] = config.getNumRegisters();
    }

    // all register values are equal to or greater than the returned lower bound
    uint32_t getRegisterValueLowerBound() const {
        return minRegisterValue;
    }

    void update(uint32_t registerIdx, R newRegisterValue) {
        assert(newRegisterValue < histogram.size());

        if (newRegisterValue > minRegisterValue) {
            uint32_t oldRegisterValue = registerValues[registerIdx];
            if (newRegisterValue > oldRegisterValue) {
                registerValues[registerIdx] = newRegisterValue;
                assert(histogram[oldRegisterValue] > 0);
                histogram[oldRegisterValue] -= 1;
                histogram[newRegisterValue] += 1;
                if (oldRegisterValue == minRegisterValue) {
                    while(histogram[minRegisterValue] == 0) minRegisterValue += 1;
                }
            }
        }
    }

    R getRegisterValue(uint32_t registerIdx) const {
        return registerValues[registerIdx];
    }

    template<typename S>
    void merge(const S& otherState) {
        for(uint32_t registerIdx = 0; registerIdx < registerValues.size(); registerIdx+=1) {
            update(registerIdx, otherState.getRegisterValue(registerIdx));
        }
    }

    std::vector<std::pair<R, uint64_t>> getHistogram() const {
        return createHistogram(registerValues);
    }

};

/*
template<typename R>
class RegistersWithLowerBound1{
    std::vector<R> registerValues;
    R limitRegisterValue;
    R registerValueLowerBound;
    uint32_t numRegistersSmallerThanOrEqualToLimit;
    static constexpr double quantileForNextLimit = 0.1;
    const uint32_t order;
public:

    typedef R RegValueType;

    template<typename C>
    RegistersWithLowerBound1(const C& config) :
        registerValues(config.getNumRegisters(), 0),
        limitRegisterValue(0),
        registerValueLowerBound(0),
        numRegistersSmallerThanOrEqualToLimit(config.getNumRegisters()),
        order(static_cast<uint32_t>(registerValues.size() * quantileForNextLimit)) {}

    // all register values are equal to or greater than the returned lower bound
    uint32_t getRegisterValueLowerBound() const {
        return registerValueLowerBound;
    }

    void update(uint32_t registerIdx, R newRegisterValue) {
        if (newRegisterValue > registerValueLowerBound) {
            uint32_t oldRegisterValue = registerValues[registerIdx];
            if (newRegisterValue > oldRegisterValue) {
                registerValues[registerIdx] = newRegisterValue;
                if (oldRegisterValue <= limitRegisterValue && newRegisterValue > limitRegisterValue) {
                    numRegistersSmallerThanOrEqualToLimit -= 1;
                    if (numRegistersSmallerThanOrEqualToLimit == 0) {
                        std::vector<R> registerValuesCopy = registerValues;
                        const auto itMid =  registerValuesCopy.begin() + order;
                        assert(itMid != registerValuesCopy.end());
                        std::nth_element(registerValuesCopy.begin(),itMid,registerValuesCopy.end());
                        registerValueLowerBound = limitRegisterValue + 1;
                        limitRegisterValue = *itMid;
                        numRegistersSmallerThanOrEqualToLimit = order;
                        for(auto it = itMid; it != registerValuesCopy.end() && (*it)== limitRegisterValue; ++it) {
                            numRegistersSmallerThanOrEqualToLimit += 1;
                        }
                    }
                }
            }
        }
    }

    R getRegisterValue(uint32_t registerIdx) const {
        return registerValues[registerIdx];
    }

    template<typename S>
    void merge(const S& otherState) {
        for(uint32_t registerIdx = 0; registerIdx < registerValues.size(); registerIdx+=1) {
            update(registerIdx, otherState.getRegisterValue(registerIdx));
        }
    }

    std::vector<std::pair<R, uint64_t>> getHistogram() const {
        return createHistogram(registerValues);
    }
};
*/

template<typename R>
class RegistersWithLowerBound{
    std::vector<R> registerValues;
    R limitRegisterValue;
    R registerValueLowerBound;
    uint32_t numRegistersSmallerThanOrEqualToLimit;
    const uint64_t limitIncrement;
public:

    typedef R RegValueType;

    template<typename C>
    RegistersWithLowerBound(const C& config) :
        registerValues(config.getNumRegisters(), 0),
        limitRegisterValue(0),
        registerValueLowerBound(0),
        numRegistersSmallerThanOrEqualToLimit(config.getNumRegisters()),
        limitIncrement(config.getSearchIncrement()) {
        assert(limitIncrement > 0);
    }

    // all register values are equal to or greater than the returned lower bound
    uint32_t getRegisterValueLowerBound() const {
        return registerValueLowerBound;
    }

    void update(uint32_t registerIdx, R newRegisterValue) {
        if (newRegisterValue > registerValueLowerBound) {
            uint32_t oldRegisterValue = registerValues[registerIdx];
            if (newRegisterValue > oldRegisterValue) {
                registerValues[registerIdx] = newRegisterValue;
                if (oldRegisterValue <= limitRegisterValue && newRegisterValue > limitRegisterValue) {
                    numRegistersSmallerThanOrEqualToLimit -= 1;
                    while (numRegistersSmallerThanOrEqualToLimit == 0) {
                        registerValueLowerBound = limitRegisterValue + 1;
                        limitRegisterValue = std::min(limitRegisterValue + limitIncrement, static_cast<uint64_t>(std::numeric_limits<R>::max()));
                        numRegistersSmallerThanOrEqualToLimit = std::count_if(registerValues.begin(), registerValues.end(), [&](R v){return v <= limitRegisterValue;});
                    }
                }
            }
        }
    }

    R getRegisterValue(uint32_t registerIdx) const {
        return registerValues[registerIdx];
    }

    template<typename S>
    void merge(const S& otherState) {
        for(uint32_t registerIdx = 0; registerIdx < registerValues.size(); registerIdx+=1) {
            update(registerIdx, otherState.getRegisterValue(registerIdx));
        }
    }

    std::vector<std::pair<R, uint64_t>> getHistogram() const {
        return createHistogram(registerValues);
    }
};

template<typename C> class HyperLogLog;
template<typename C> class GeneralizedHyperLogLog;
template<typename C> class SetSketch1;
template<typename C> class SetSketch2;

template<typename S>
class GeneralizedHyperLogLogConfig {
public:
    typedef WyrandBitStream BitStreamType;
    typedef typename S::RegValueType RegValueType;
    typedef S RegStateType;
    typedef GeneralizedHyperLogLog<GeneralizedHyperLogLogConfig> SketchType;
private:
    const uint32_t numRegisters;
    const double base;
    const uint64_t seed;
    const uint64_t q;
    const double a;
    const CardinalityEstimator cardinalityEstimator;
    const Mapping mapping;
public:

    GeneralizedHyperLogLogConfig(uint32_t numRegisters, double base, uint64_t q, uint64_t seed) :
            numRegisters(numRegisters),
            base(base),
            seed(seed),
            q(q),
            a(1./numRegisters),
            cardinalityEstimator(q, 1. / numRegisters, base, numRegisters),
            mapping(base, q) {

        assert(numRegisters > 0);
        assert(base > 1);
        assert(std::numeric_limits<RegValueType>::max() > q);
    }

    bool operator==(const GeneralizedHyperLogLogConfig& config) const {
        if (numRegisters != config.numRegisters) return false;
        if (base != config.base) return false;
        if (seed != config.seed) return false;
        if (q != config.q) return false;
        return true;
    }

    uint32_t getNumRegisters() const {return numRegisters;}

    BitStreamType getBitStream(uint64_t x) const {return BitStreamType(x, seed);}

    double getBase() const {return base;}

    uint64_t getQ() const {return q;}

    double getA() const {return a;}

    const CardinalityEstimator& getEstimator() const {return cardinalityEstimator;}

    GeneralizedHyperLogLog<GeneralizedHyperLogLogConfig<S>> create() const {return GeneralizedHyperLogLog(*this);}

    std::string getName() const {return "GeneralizedHyperLogLog";}

    uint64_t getSeed() const {return seed;}

    const Mapping& getMapping() const {return mapping;}

    uint64_t getSearchIncrement() const {
        return mapping.getSearchIncrement();
    }
};


template<typename S>
class HyperLogLogConfig {
public:
    typedef typename S::RegValueType RegValueType;
    typedef S RegStateType;
    typedef HyperLogLog<HyperLogLogConfig> SketchType;
private:
    const uint32_t numRegisters;
    const uint64_t p;
    const uint64_t q;
    const double a;
    const CardinalityEstimator cardinalityEstimator;
public:

    HyperLogLogConfig(uint64_t p, uint64_t q) :
            numRegisters(UINT32_C(1) << p),
            p(p),
            q(q),
            a(1./numRegisters),
            cardinalityEstimator(q, a, 2., numRegisters) {

        assert(p + q <= 64);
        assert(numRegisters > 0);
        assert(std::numeric_limits<RegValueType>::max() > q);
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

    const CardinalityEstimator& getEstimator() const {return cardinalityEstimator;}

    HyperLogLog<HyperLogLogConfig<S>> create() const {return HyperLogLog(*this);}

    std::string getName() const {return "HyperLogLog";}

    uint64_t getSeed() const {return 0;}

    uint64_t getSearchIncrement() const {
        return UINT64_C(1);
    }

};

template<typename C>
class BaseSketch {
protected:
    const C& config;
    typename C::RegStateType state;
public:

    BaseSketch(const C& config) : config(config), state(config) {}

    void merge(const BaseSketch<C>& other) {
        assert(config == other.getConfig());
        state.merge(other.state);
    }

    const C& getConfig() const {
        return config;
    }

    double estimateCardinalityML(bool useRangeCorrection) const {
        return this->config.getEstimator().estimateCardinalityML(this->state.getHistogram(), useRangeCorrection);
    }

    double estimateCardinalitySimple(bool useRangeCorrection) const {
       return this->config.getEstimator().estimateCardinalitySimple(this->state.getHistogram(), useRangeCorrection);
    }

    const typename C::RegStateType& getState() const {
        return state;
    }
};

template<typename S>
JointEstimationResult estimateJointML(const S& sketch1, const S& sketch2, bool useRangeCorrection) {
    assert(sketch1.getConfig() == sketch2.getConfig());
    return sketch1.getConfig().getEstimator().estimateJointML(sketch1.getState(), sketch2.getState(), useRangeCorrection);
}

template<typename S>
JointEstimationResult estimateJointSimple(const S& sketch1, const S& sketch2, bool useRangeCorrection) {
    assert(sketch1.getConfig() == sketch2.getConfig());
    return sketch1.getConfig().getEstimator().estimateJointSimple(sketch1.getState(), sketch2.getState(), useRangeCorrection);
}

template<typename S>
JointEstimationResult estimateJointInclExcl(const S& sketch1, const S& sketch2, bool useRangeCorrection) {
    assert(sketch1.getConfig() == sketch2.getConfig());
    return sketch1.getConfig().getEstimator().estimateJointInclExcl(sketch1.getState(), sketch2.getState(), useRangeCorrection);
}

template<typename S>
std::pair<double,double> estimateJaccardSimilarityUsingEqualRegisters(const S& sketch1, const S& sketch2) {
    assert(sketch1.getConfig() == sketch2.getConfig());
    return sketch1.getConfig().getEstimator().estimateJaccardSimilarityUsingEqualRegisters(sketch1.getState(), sketch2.getState());
}


template<typename C>
class GeneralizedHyperLogLog : public BaseSketch<C> {
public:

    GeneralizedHyperLogLog(const C& config) : BaseSketch<C>{config} {}

    void add(uint64_t d) {
        auto bitstream = this->config.getBitStream(d);
        double x = getUniformDouble(bitstream);
        auto kLow = this->state.getRegisterValueLowerBound();
        auto k = this->config.getMapping().map(kLow, x);
        if (k == kLow) return;
        uint32_t registerIdx = getUniformLemire(this->config.getNumRegisters(), bitstream);
        this->state.update(registerIdx, k);
    }
};

template<typename C>
class HyperLogLog : public BaseSketch<C> {
public:

    HyperLogLog(const C& config) : BaseSketch<C>{config} {}

    void add(uint64_t d) {

        if (((UINT64_C(0xFFFFFFFFFFFFFFFF) << this->state.getRegisterValueLowerBound()) | d) == UINT64_C(0xFFFFFFFFFFFFFFFF)) {
            uint32_t registerIdx = static_cast<uint32_t>(d >> (64 - this->config.getP()));
            uint8_t k = 1;
            while(k <= this->config.getQ() && (d & 1)) {
                k += 1;
                d >>= 1;
            }
            this->state.update(registerIdx, k);
        }
    }
};


template<typename S>
class SetSketchConfig1 {
public:
    typedef WyrandBitStream BitStreamType;
    typedef typename S::RegValueType RegValueType;
    typedef S RegStateType;
    typedef SetSketch1<SetSketchConfig1> SketchType;
private:
    const uint32_t numRegisters;
    double base;
    const uint64_t seed;
    const uint64_t q;
    const double a;
    const CardinalityEstimator cardinalityEstimator;
    std::vector<double> factors;
    const Mapping mapping;
public:

    SetSketchConfig1(uint32_t numRegisters, double base, double a, uint64_t q, uint64_t seed) :
            numRegisters(numRegisters),
            base(base),
            seed(seed),
            q(q),
            a(a),
            cardinalityEstimator(q, a, base, numRegisters),
            factors(numRegisters),
            mapping(base, q) {

        assert(numRegisters > 0);
        assert(base > 1);
        assert(std::numeric_limits<RegValueType>::max() > q);

        for(uint32_t i = 0; i < numRegisters; ++i) factors[i] = 1./(a * (numRegisters - i));
    }

    bool operator==(const SetSketchConfig1& config) const {
        if (numRegisters != config.numRegisters) return false;
        if (base != config.base) return false;
        if (a != config.a) return false;
        if (seed != config.seed) return false;
        if (q != config.q) return false;
        return true;
    }

    uint32_t getNumRegisters() const {return numRegisters;}

    BitStreamType getBitStream(uint64_t x) const {return BitStreamType(x, seed);}

    double getBase() const {return base;}

    uint64_t getQ() const {return q;}

    double getA() const {return a;}

    double getFactor(uint32_t i) const {return factors[i];}

    const CardinalityEstimator& getEstimator() const {return cardinalityEstimator;}

    SetSketch1<SetSketchConfig1<S>> create() const {return SetSketch1(*this);}

    std::string getName() const {return "SetSketch1";}

    uint64_t getSeed() const {return seed;}

    const Mapping& getMapping() const {return mapping;}

    uint64_t getSearchIncrement() const {
        return mapping.getSearchIncrement();
    }
};


template<typename C>
class SetSketch1 : public BaseSketch<C> {
    PermutationStream permutationStream;
public:

    SetSketch1(const C& config) : BaseSketch<C>{config} {}

    SetSketch1(const SetSketch1& sketch) : BaseSketch<C>(sketch) {}

    void add(uint64_t d) {
        auto bitstream = this->config.getBitStream(d);
        permutationStream.reset(this->config.getNumRegisters());
        double x = 0;
        for(uint32_t i = 0; i < this->config.getNumRegisters(); ++i) {
            x += ziggurat::getExponential(bitstream) * this->config.getFactor(i);
            auto kLow = this->state.getRegisterValueLowerBound();
            auto k = this->config.getMapping().map(kLow, x);
            if (k == kLow) return;
            uint32_t registerIdx = permutationStream.next(bitstream);
            this->state.update(registerIdx, k);
        }
    }
};

template<typename S>
class SetSketchConfig2 {
public:
    typedef WyrandBitStream BitStreamType;
    typedef typename S::RegValueType RegValueType;
    typedef S RegStateType;
    typedef SetSketch2<SetSketchConfig2> SketchType;
private:
    const uint32_t numRegisters;
    const double base;
    const uint64_t seed;
    const uint64_t q;
    const double a;
    const CardinalityEstimator cardinalityEstimator;
    std::vector<double> gammaTimesAInv;
    std::vector<TruncatedExponentialDistribution> truncatedExponentialDistributions;
    const double aInv;
    const Mapping mapping;
public:

    SetSketchConfig2(uint32_t numRegisters, double base, double a, uint64_t q, uint64_t seed) :
            numRegisters(numRegisters),
            base(base),
            seed(seed),
            q(q),
            a(a),
            cardinalityEstimator(q, a, base, numRegisters),
            gammaTimesAInv(numRegisters),
            truncatedExponentialDistributions(numRegisters - 1),
            aInv(1. / a),
            mapping(base, q) {

        assert(numRegisters > 0);
        assert(base > 1);
        assert(std::numeric_limits<RegValueType>::max() > q);

        for(uint32_t i = 0; i < numRegisters - 1; ++i) truncatedExponentialDistributions[i] = TruncatedExponentialDistribution(std::log1p(1./static_cast<double>(numRegisters - i - 1)));
        for(uint32_t i = 0; i < numRegisters; ++i) gammaTimesAInv[i] =
            std::log1p(static_cast<double>(i)/static_cast<double>(numRegisters - i)) * aInv;
    }

    bool operator==(const SetSketchConfig2& config) const {
        if (numRegisters != config.numRegisters) return false;
        if (base != config.base) return false;
        if (a != config.a) return false;
        if (seed != config.seed) return false;
        if (q != config.q) return false;
        return true;
    }

    uint32_t getNumRegisters() const {return numRegisters; }

    BitStreamType getBitStream(uint64_t x) const {return BitStreamType(x, seed);}

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

    const CardinalityEstimator& getEstimator() const {return cardinalityEstimator;}

    SetSketch2<SetSketchConfig2<S>> create() const {return SetSketch2(*this);}

    uint64_t getSeed() const {return seed;}

    std::string getName() const {return "SetSketch2";}

    const Mapping& getMapping() const {return mapping;}

    uint64_t getSearchIncrement() const {
        return mapping.getSearchIncrement();
    }

};

template<typename C>
class SetSketch2 : public BaseSketch<C> {
    PermutationStream permutationStream;
public:

    SetSketch2(const C& config) : BaseSketch<C>{config} {}

    SetSketch2(const SetSketch2& sketch) : BaseSketch<C>(sketch) {}

    void add(uint64_t d) {
        auto bitstream = this->config.getBitStream(d);
        permutationStream.reset(this->config.getNumRegisters());
        const auto& c = this->config;
        const uint32_t m = c.getNumRegisters();

        for(uint32_t i = 0; i < m; ++i) {
            auto kLow = this->state.getRegisterValueLowerBound();
            if (!this->config.getMapping().isRelevant(kLow, c.getGammaTimesAInv(i))) break;
            double x;
            if (i < m - 1) {
                x = c.getGammaTimesAInv(i) + (c.getGammaTimesAInv(i+1) - c.getGammaTimesAInv(i)) * c.getTruncatedExponentialDistribution(i)(bitstream);
            } else {
                x = c.getGammaTimesAInv(m-1) + c.getAInv() * ziggurat::getExponential(bitstream);
            }
            auto k = this->config.getMapping().map(kLow, x);
            if (k == kLow) return;
            uint32_t registerIdx = permutationStream.next(bitstream);
            this->state.update(registerIdx, k);
        }
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
    os << "seed=" << std::hex << std::setfill('0') << std::setw(16) << config.getSeed() << std::dec << ";";
    os.flags( f );
}

#endif // _SKETCH_HPP_