# Copyright (c) 2012-2021 Dynatrace LLC. All rights reserved.
#
# This software and associated documentation files (the "Software")
# are being made available by Dynatrace LLC for purposes of
# illustrating the implementation of certain algorithms which have
# been published by Dynatrace LLC. Permission is hereby granted,
# free of charge, to any person obtaining a copy of the Software,
# to view and use the Software for internal, non-productive,
# non-commercial purposes only – the Software may not be used to
# process live data or distributed, sublicensed, modified and/or
# sold either alone or as part of or in combination with any other
# software.
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from scipy.stats import chisquare
from scipy.stats import kstest
from scipy.stats import binom_test
from numpy import expm1

def testUniformIntegerDistribution(fileName, n, significanceLevel):
    with open(fileName) as f:
        content = f.readlines()
    values = [int(x) for x in content]
    histogram = [0] * n
    for v in values:
        histogram[v] += 1

    result = chisquare(histogram)
    pValue = result[1]

def testUniformDoubleDistribution(fileName, significanceLevel):
    with open(fileName) as f:
        content = f.readlines()
    values = [float(x) for x in content]
    _, pValue = kstest(values, "uniform")
    assert(pValue > significanceLevel)

def testUniformDoubleHalfDistribution(fileName, significanceLevel):
    with open(fileName) as f:
        content = f.readlines()
    values = [2*float(x) for x in content]
    _, pValue = kstest(values, "uniform")
    assert(pValue > significanceLevel)

def testExponentialDistribution(fileName, significanceLevel):
    with open(fileName) as f:
        content = f.readlines()
    values = [float(x) for x in content]
    _, pValue = kstest(values, "expon")
    assert(pValue > significanceLevel)


def testBernoulliDistribution(fileName, successProbability, significanceLevel):
    with open(fileName) as f:
        content = f.readlines()
    values = [int(x) for x in content]
    x = sum(values)
    pValue = binom_test(x, len(values), successProbability)
    assert(pValue > significanceLevel)

def cdfTruncatedExp(rate, x):
    if (rate > 0):
        return expm1(-x * rate) / expm1(-rate)
    else:
        return x

def testTruncatedExponentialDistribution(fileName, rate, significanceLevel):
    with open(fileName) as f:
        content = f.readlines()
        values = [float(x) for x in content]

        cdf = lambda x : cdfTruncatedExp(rate, x)

        _, pValue = kstest(values, cdf)
        assert(pValue > significanceLevel)


significanceLevel = 0.01

testUniformIntegerDistribution("data/uniformLemire3.txt", 3, significanceLevel)
testUniformIntegerDistribution("data/uniformLemire11.txt", 11, significanceLevel)
testUniformIntegerDistribution("data/uniformLemire29.txt", 29, significanceLevel)
testUniformIntegerDistribution("data/uniformLemire256.txt", 256, significanceLevel)
testUniformIntegerDistribution("data/uniformLumbroso3.txt", 3, significanceLevel)
testUniformIntegerDistribution("data/uniformLumbroso11.txt", 11, significanceLevel)
testUniformIntegerDistribution("data/uniformLumbroso29.txt", 29, significanceLevel)
testUniformIntegerDistribution("data/uniformLumbroso256.txt", 256, significanceLevel)
testUniformIntegerDistribution("data/intPow3.txt", pow(2, 3), significanceLevel)
testUniformIntegerDistribution("data/intPow8.txt", pow(2, 8), significanceLevel)

testUniformDoubleDistribution("data/uniformDouble.txt", significanceLevel)
testUniformDoubleHalfDistribution("data/uniformDoubleHalf.txt", significanceLevel)

testExponentialDistribution("data/expStandard.txt", significanceLevel)
testExponentialDistribution("data/expZiggurat.txt", significanceLevel)

testBernoulliDistribution("data/boolean.txt", 0.5, significanceLevel)
testBernoulliDistribution("data/bernoulliReal0_2.txt", 0.2, significanceLevel)
testBernoulliDistribution("data/bernoulliRatio1_3.txt", 1./3., significanceLevel)

testTruncatedExponentialDistribution("data/truncatedExp0.txt", 0, significanceLevel)
testTruncatedExponentialDistribution("data/truncatedExp0_1.txt", 0.1, significanceLevel)
testTruncatedExponentialDistribution("data/truncatedExp0_5.txt", 0.5, significanceLevel)
testTruncatedExponentialDistribution("data/truncatedExp1.txt", 1, significanceLevel)
testTruncatedExponentialDistribution("data/truncatedExp2.txt", 2, significanceLevel)
