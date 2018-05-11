#ifndef LINEAROPTICALTRANSFORM_H_INCLUDED
#define LINEAROPTICALTRANSFORM_H_INCLUDED

#include <iostream>
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

        template <typename T>
        void printVec(std::vector<T>& a);

        bool iterate(int bitstring[],int& j,int& k,bool& ending,bool& sign);
        double numbPermutations(int& i);
        double doublefactorial(int x);
        void rysersAlgorithm(Eigen::MatrixXcd& U,int& i,double& parallelMutualInformation);
        void permutationAlgorithm(Eigen::MatrixXcd& U,int& i,double& parallelMutualInformation);
        void setmVec(std::vector<int>& m, std::vector<int>& n);

        inline double boolPow(bool& x);

        Eigen::Matrix4d bellStates;

};

inline double LinearOpticalTransform::boolPow(bool& x){

    return -1 + 2 * x;

}

#endif // LINEAROPTICALTRANSFORM_H_INCLUDED

