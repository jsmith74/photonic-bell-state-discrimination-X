#ifndef LINEAROPTICALTRANSFORM_H_INCLUDED
#define LINEAROPTICALTRANSFORM_H_INCLUDED

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <iomanip>
#include "GrayCode.h"

class LinearOpticalTransform{

    public:

        LinearOpticalTransform();
        void initializeCircuit(Eigen::MatrixXi& inBasis, Eigen::MatrixXi& outBasis);
        void setMutualInformation(Eigen::MatrixXcd& U);

        double mutualInformation;

    private:

        std::vector< std::vector<int> > n,m,nPrime,mPrime;
        std::vector<double> factorial;
        std::vector<bool> useRysers;
        int photons;
        GrayCode graycode;

        template <typename T>
        void printVec(std::vector<T>& a);

        double numbPermutations(int& i);
        double doublefactorial(int x);
        void permutationAlgorithm(Eigen::MatrixXcd& U,int& i);
        void setmVec(std::vector<int>& m, std::vector<int>& n);
        void rysersAlgorithm(Eigen::MatrixXcd& U,int& i);
        inline double boolPow(bool& x);

        void setParallelGrid(int micRatio);
        void checkThreadsAndProcs();
        void setTotalTerms();
        void allocateWorkToThreads();
        void printParallelGrid();

        int numThreadsCPU, numCoprocessors, numThreadsPhi, totalTerms, coprocessorTerms, CPUTerms,
            coprocessorTermsPerThread, CPUTermsPerThread;

};

inline double LinearOpticalTransform::boolPow(bool& x){

    return -1 + 2 * x;

}

#endif // LINEAROPTICALTRANSFORM_H_INCLUDED

