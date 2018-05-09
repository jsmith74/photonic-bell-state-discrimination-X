/*! \file main.cpp
 *  \brief This is a demonstration on how to use the LinearOpticalTransform object.
 *
 *     The LinearOpticalTransform object constructs a quantum evolution operator from the unitary matrix
 *     describing a linear optical quantum circuit.
 *
 *
 *     For more information, refer to:
 *      https://arxiv.org/abs/1711.01319
 *
 *
 *     Here, we construct the evolution operator A(U) for a simple Mach-Zehnder interferometer; we apply a phase shift
 *     of pi/3 to the first optical mode, and then apply a 50-50 beam splitter between modes 1 and 2.
 *
 *
 * \author Jake Smith <jsmith74@tulane.edu>
 */

#include "LinearOpticalTransform.h"
#define PI 3.141592653589793
#include "omp.h"
#include <unsupported/Eigen/MatrixFunctions>


void printState(Eigen::VectorXcd& vec,Eigen::MatrixXi& basis);
void setToFullHilbertSpace(const int& subPhotons, const int& subModes,Eigen::MatrixXi& nv);
void setToRandomBasisStates(Eigen::MatrixXi& basis,int photons,int modes,int basisDim);


int main(){

    /** Establish number of photons and modes and input and output Fock basis  */

    int ancillaPhotons = 6;
    int ancillaModes = 8;

    int photons = 2 + ancillaPhotons;
    int modes = 4 + ancillaModes;

    Eigen::MatrixXcd H = Eigen::MatrixXcd::Random(modes,modes);

    H += H.conjugate().transpose().eval();

    H *= std::complex<double>(0.0,1.0);

    Eigen::MatrixXcd U = H.exp();

    std::cout << U << std::endl << std::endl;

    LinearOpticalTransform LOCircuit;

    Eigen::MatrixXi inBasis, outBasis;

    inBasis = Eigen::MatrixXi::Zero(4,modes);

    for(int i=0;i<4;i++) for(int j=0;j<ancillaPhotons;j++) inBasis(i,j) = 1;

    inBasis(0,ancillaModes) = 1;      inBasis(0,ancillaModes+2) = 1;
    inBasis(1,ancillaModes+1) = 1;    inBasis(1,ancillaModes+3) = 1;
    inBasis(2,ancillaModes) = 1;      inBasis(2,ancillaModes+3) = 1;
    inBasis(3,ancillaModes+1) = 1;    inBasis(3,ancillaModes+2) = 1;

    setToFullHilbertSpace(photons,modes,outBasis);

    LOCircuit.initializeCircuit(inBasis,outBasis);

    double startTime = omp_get_wtime();

    //for(int i=0;i<4;i++)
        LOCircuit.setMutualInformation(U);

    double endTime = omp_get_wtime();

    std::cout << "Running time: " << endTime - startTime << std::endl;
    std::cout << "H(X:Y): " << std::setprecision(16) << 2.0 - 0.25 * LOCircuit.mutualInformation << std::endl;

    return 0;

}

void setToRandomBasisStates(Eigen::MatrixXi& basis,int photons,int modes,int basisDim){

    basis = Eigen::MatrixXi::Zero(basisDim,modes);

    for(int i=0;i<basisDim;i++) for(int j=0;j<modes;j++){

        basis(i,j) = rand() % ( 1 + photons - basis.row(i).sum() );

	if(j==modes-1) basis(i,j) += photons - basis.row(i).sum();

    }

    return;

}

inline double doublefactorial(int x){

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

inline int g(const int& n,const int& m){
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

void setToFullHilbertSpace(const int& subPhotons, const int& subModes,Eigen::MatrixXi& nv){

    if(subPhotons==0 && subModes == 0){

        nv.resize(0,0);

        return;

    }

    int markers = subPhotons + subModes - 1;
    int myints[markers];
    int i = 0;
    while(i<subPhotons){
        myints[i]=1;
        i++;
    }
    while(i<markers){
        myints[i]=0;
        i++;
    }
    nv = Eigen::MatrixXi::Zero(g(subPhotons,subModes),subModes);
    i = 0;
    int j,k = 0;
    do {
        j = 0;
        k = 0;
        while(k<markers){
        if(myints[k]==1){
            nv(i,j)=nv(i,j)+1;
        }
        else if(myints[k]==0){
            j++;
        }

        k++;
        }
        i++;
    } while ( std::prev_permutation(myints,myints+markers) );
    return;;
}


void printState(Eigen::VectorXcd& vec,Eigen::MatrixXi& basis){

    for(int i=0;i<vec.size();i++){

        std::cout << vec(i) << " * |" << basis.row(i) << ">\n";

    }

    std::cout << std::endl;

    return;

}
