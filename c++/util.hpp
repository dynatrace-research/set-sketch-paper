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

#ifndef _UTIL_HPP_
#define _UTIL_HPP_

#include <vector>
#include <cmath>
#include <cassert>

template<typename F>
double calculateMSE(const F& estimatedValues, uint64_t size, double trueValue) {
    double squaredErrorSum = 0;
    for(uint64_t i = 0; i < size; ++i) {
        squaredErrorSum += pow(estimatedValues(i) - trueValue, 2);  
    }
    double result = squaredErrorSum / size;
    return result;
}

double calculateMSE(const std::vector<double>& estimatedValues, double trueValue) {
    return calculateMSE([&estimatedValues](uint64_t idx){return estimatedValues[idx];}, estimatedValues.size(), trueValue);
}

template<typename F>
double calculateMean(const F& estimatedValues, uint64_t size) {
    double sum = 0;
    for(uint64_t i = 0; i < size; ++i) {
        sum += estimatedValues(i);  
    }
    
    double result = sum / size;
    return result;
}

double calculateMean(const std::vector<double>& estimatedValues) {
    return calculateMean([&estimatedValues](uint64_t idx){return estimatedValues[idx];}, estimatedValues.size());
}


std::vector<uint64_t> getCardinalities(uint64_t max, double relativeIncrement) {
    assert(relativeIncrement > 0);
    std::vector<uint64_t> cardinalities;
    for(uint64_t c = max; c > 0; c = std::min(c-1, static_cast<uint64_t>(std::ceil(c / (1. + relativeIncrement))))) cardinalities.push_back(c);
    cardinalities.push_back(0);
    reverse(cardinalities.begin(), cardinalities.end());
    return cardinalities;
}


#endif // _SKETCH_HPP_