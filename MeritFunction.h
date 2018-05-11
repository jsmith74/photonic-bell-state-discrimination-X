#ifndef MERITFUNCTION_H_INCLUDED
#define MERITFUNCTION_H_INCLUDED

#include "LinearOpticalTransform.h"

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <unsupported/Eigen/MatrixFunctions>
#include <omp.h>

class MeritFunction{

    public:

        MeritFunction();
        void setMeritFunction(int intParam);
        double f(Eigen::VectorXd& position);
        int funcDimension;
        void printReport(Eigen::VectorXd& position);
        Eigen::VectorXd setInitialPosition();
        double entropyMonitor();

    private:

        LinearOpticalTransform LOCircuit;
        Eigen::MatrixXcd U;

        void setAntiHermitian( Eigen::MatrixXcd& H,Eigen::VectorXd& position );
        void setPosition(Eigen::MatrixXcd& U, Eigen::VectorXd& position);
        void shiftUToZeroSolution(Eigen::VectorXd& position);

        void setToFullHilbertSpace(const int& subPhotons, const int& subModes,Eigen::MatrixXi& nv);
        inline int g(const int& n,const int& m);
        inline double doublefactorial(int x);
};


inline double MeritFunction::doublefactorial(int x){

    assert(x < 171);

    double total=1.0;
    if (x>=0){
        for(int i=x;i>0;i--){
            total=i*total;
        }
    }
    else{
        std::cout << "invalid factorial" << std::endl;
        total=-1;
    }
    return total;
}

inline int MeritFunction::g(const int& n,const int& m){
    if(n==0 && m==0){
        return 0;
    }
    else if(n==0 && m>0){
        return 1;
    }

    else{
        return (int)(doublefactorial(n+m-1)/(doublefactorial(n)*doublefactorial(m-1))+0.5);
    }
}

#endif // MERITFUNCTION_H_INCLUDED
