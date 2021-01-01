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

#include "exponential_distribution.hpp"
#include "bitstream_random.hpp"

#include <iostream>
#include <iomanip>
#include <string>
#include <functional>
#include <fstream>
#include <chrono>
#include <vector>

using namespace std;

template<typename V>
void generateAndWriteRandomNumbers(string fileName, function<V(WyrandBitStream&)> generator) {
    ofstream out(fileName);

    uint64_t seed = UINT64_C(0xa9142ff6f733a101);

    uint64_t numOffsets = 64;
    uint64_t numNumbers = 10000;

    vector<WyrandBitStream> bitStreams;
    for(uint64_t offset = 0; offset < numOffsets; ++offset) {
        WyrandBitStream bitStream(offset, seed);
        for(uint64_t j = 0; j < offset; ++j) {
            bitStream();
        }
        bitStreams.emplace_back(move(bitStream));
    }

    std::vector<V> values(numNumbers * numOffsets);
    chrono::steady_clock::time_point tStart = chrono::steady_clock::now();
    uint64_t counter = 0;
    for(uint64_t offset = 0; offset < numOffsets; ++offset) {
        WyrandBitStream& bitStream = bitStreams[offset];
        for(uint64_t i = 0; i < numNumbers; ++i) {
            values[counter] = generator(bitStream);
            counter += 1;
        }
    }
    chrono::steady_clock::time_point tEnd = chrono::steady_clock::now();

    double calculationTimeNanos = (chrono::duration_cast<chrono::duration<double>>(tEnd - tStart).count() / counter) * 1e9;

    cout << fileName << " " << calculationTimeNanos << "ns" << endl;

    for(auto& v : values) {
        out << setprecision(std::numeric_limits<V>::max_digits10) << v << endl;
    }
}

int main(int argc, char* argv[]) {
    assert(argc == 2);
    string outputFolder = argv[1];

    const TruncatedExponentialDistribution truncatedExponentialDistribution0(0);
    const TruncatedExponentialDistribution truncatedExponentialDistribution0_1(0.1);
    const TruncatedExponentialDistribution truncatedExponentialDistribution0_5(0.5);
    const TruncatedExponentialDistribution truncatedExponentialDistribution1(1);
    const TruncatedExponentialDistribution truncatedExponentialDistribution2(2);

    function<uint64_t(WyrandBitStream&)> boolGenerator = [](WyrandBitStream& bitStream) {return bitStream();};
    function<uint64_t(WyrandBitStream&)> uniformLemire3Generator =[](WyrandBitStream& bitStream) {return getUniformLemire(3, bitStream);};
    function<uint64_t(WyrandBitStream&)> uniformLemire11Generator =[](WyrandBitStream& bitStream) {return getUniformLemire(11, bitStream);};
    function<uint64_t(WyrandBitStream&)> uniformLemire29Generator =[](WyrandBitStream& bitStream) {return getUniformLemire(29, bitStream);};
    function<uint64_t(WyrandBitStream&)> uniformLemire256Generator =[](WyrandBitStream& bitStream) {return getUniformLemire(256, bitStream);};
    function<uint64_t(WyrandBitStream&)> uniformLumbroso3Generator =[](WyrandBitStream& bitStream) {return getUniformLumbroso(3, bitStream);};
    function<uint64_t(WyrandBitStream&)> uniformLumbroso11Generator =[](WyrandBitStream& bitStream) {return getUniformLumbroso(11, bitStream);};
    function<uint64_t(WyrandBitStream&)> uniformLumbroso29Generator =[](WyrandBitStream& bitStream) {return getUniformLumbroso(29, bitStream);};
    function<uint64_t(WyrandBitStream&)> uniformLumbroso256Generator =[](WyrandBitStream& bitStream) {return getUniformLumbroso(256, bitStream);};
    function<uint64_t(WyrandBitStream&)> intPow3Generator =[](WyrandBitStream& bitStream) {return getUniformPow2(3, bitStream);};
    function<uint64_t(WyrandBitStream&)> intPow8Generator =[](WyrandBitStream& bitStream) {return getUniformPow2(8, bitStream);};
    function<double(WyrandBitStream&)> expZigguratGenerator =[](WyrandBitStream& bitStream) {return ziggurat::getExponential(bitStream);};
    function<double(WyrandBitStream&)> expStandardGenerator =[](WyrandBitStream& bitStream) {return getExponential1(bitStream);};
    function<double(WyrandBitStream&)> uniformDoubleGenerator =[](WyrandBitStream& bitStream) {return getUniformDouble(bitStream);};
    function<double(WyrandBitStream&)> uniformDoubleHalfGenerator =[](WyrandBitStream& bitStream) {return getUniformDoubleHalf(bitStream);};
    function<uint64_t(WyrandBitStream&)> bernoulliReal0_2Generator =[](WyrandBitStream& bitStream) {return getBernoulli(0.2, bitStream);};
    function<uint64_t(WyrandBitStream&)> bernoulliRational1_3Generator =[](WyrandBitStream& bitStream) {return getBernoulli(1, 3, bitStream);};    
    function<double(WyrandBitStream&)> truncatedExp0Generator =[&truncatedExponentialDistribution0](WyrandBitStream& bitStream) {return truncatedExponentialDistribution0(bitStream);};
    function<double(WyrandBitStream&)> truncatedExp0_1Generator =[&truncatedExponentialDistribution0_1](WyrandBitStream& bitStream) {return truncatedExponentialDistribution0_1(bitStream);};
    function<double(WyrandBitStream&)> truncatedExp0_5Generator =[&truncatedExponentialDistribution0_5](WyrandBitStream& bitStream) {return truncatedExponentialDistribution0_5(bitStream);};
    function<double(WyrandBitStream&)> truncatedExp1Generator =[&truncatedExponentialDistribution1](WyrandBitStream& bitStream) {return truncatedExponentialDistribution1(bitStream);};
    function<double(WyrandBitStream&)> truncatedExp2Generator =[&truncatedExponentialDistribution2](WyrandBitStream& bitStream) {return truncatedExponentialDistribution2(bitStream);};

    generateAndWriteRandomNumbers(outputFolder + string("/boolean.txt"), boolGenerator);
    generateAndWriteRandomNumbers(outputFolder + string("/uniformLemire3.txt"), uniformLemire3Generator);
    generateAndWriteRandomNumbers(outputFolder + string("/uniformLemire11.txt"), uniformLemire11Generator);
    generateAndWriteRandomNumbers(outputFolder + string("/uniformLemire29.txt"), uniformLemire29Generator);
    generateAndWriteRandomNumbers(outputFolder + string("/uniformLemire256.txt"), uniformLemire256Generator);
    generateAndWriteRandomNumbers(outputFolder + string("/uniformLumbroso3.txt"), uniformLumbroso3Generator);
    generateAndWriteRandomNumbers(outputFolder + string("/uniformLumbroso11.txt"), uniformLumbroso11Generator);
    generateAndWriteRandomNumbers(outputFolder + string("/uniformLumbroso29.txt"), uniformLumbroso29Generator);
    generateAndWriteRandomNumbers(outputFolder + string("/uniformLumbroso256.txt"), uniformLumbroso256Generator);
    generateAndWriteRandomNumbers(outputFolder + string("/intPow3.txt"), intPow3Generator);
    generateAndWriteRandomNumbers(outputFolder + string("/intPow8.txt"), intPow8Generator);
    generateAndWriteRandomNumbers(outputFolder + string("/expZiggurat.txt"), expZigguratGenerator);
    generateAndWriteRandomNumbers(outputFolder + string("/expStandard.txt"), expStandardGenerator);
    generateAndWriteRandomNumbers(outputFolder + string("/uniformDouble.txt"), uniformDoubleGenerator);
    generateAndWriteRandomNumbers(outputFolder + string("/uniformDoubleHalf.txt"), uniformDoubleHalfGenerator);
    generateAndWriteRandomNumbers(outputFolder + string("/bernoulliReal0_2.txt"), bernoulliReal0_2Generator);
    generateAndWriteRandomNumbers(outputFolder + string("/bernoulliRatio1_3.txt"), bernoulliRational1_3Generator);
    generateAndWriteRandomNumbers(outputFolder + string("/truncatedExp0.txt"), truncatedExp0Generator);
    generateAndWriteRandomNumbers(outputFolder + string("/truncatedExp0_1.txt"), truncatedExp0_1Generator);
    generateAndWriteRandomNumbers(outputFolder + string("/truncatedExp0_5.txt"), truncatedExp0_5Generator);
    generateAndWriteRandomNumbers(outputFolder + string("/truncatedExp1.txt"), truncatedExp1Generator);
    generateAndWriteRandomNumbers(outputFolder + string("/truncatedExp2.txt"), truncatedExp2Generator);

}