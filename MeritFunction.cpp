#include "MeritFunction.h"

#define PI 3.141592653589793

#define SUCCESS_MUT_ENT 1.55

#define AMPLITUDE_SCALING 1000.0

#define INITIAL_CONDITION_RANDOM_DEGREE 2000

#define ZERO_ENTRY_WEIGHT 0.1

#define START_NEAR_ZERO_SOLUTION

void MeritFunction::setMeritFunction(int intParam){

    int ancillaPhotons = 6;
    int ancillaModes = 8;

    int photons = 2 + ancillaPhotons;
    int modes = 4 + ancillaModes;

    Eigen::MatrixXi inBasis, outBasis;

    inBasis = Eigen::MatrixXi::Zero(4,modes);

    for(int i=0;i<4;i++) for(int j=0;j<ancillaPhotons;j++) inBasis(i,j) = 1;

    inBasis(0,ancillaModes) = 1;      inBasis(0,ancillaModes+2) = 1;
    inBasis(1,ancillaModes+1) = 1;    inBasis(1,ancillaModes+3) = 1;
    inBasis(2,ancillaModes) = 1;      inBasis(2,ancillaModes+3) = 1;
    inBasis(3,ancillaModes+1) = 1;    inBasis(3,ancillaModes+2) = 1;

    setToFullHilbertSpace(photons,modes,outBasis);

    LOCircuit.initializeCircuit(inBasis,outBasis);

    funcDimension = (4 + ancillaModes) * (4 + ancillaModes);

    U.resize( 4 + ancillaModes,4 + ancillaModes );

    return;

}



double MeritFunction::f(Eigen::VectorXd& position){

    setAntiHermitian( U , position );

    U = U.exp().eval();

    LOCircuit.setMutualInformation(U);

    return LOCircuit.mutualInformation;

}

double MeritFunction::entropyMonitor(){

    return 2.0 - 0.25 * LOCircuit.mutualInformation;

}


void MeritFunction::printReport(Eigen::VectorXd& position){

    setAntiHermitian( U , position );

    U = U.exp().eval();

    LOCircuit.setMutualInformation(U);

    std::ofstream outfile("resultMonitor.dat",std::ofstream::app);

    outfile << "H(X:Y): " << std::setprecision(16) << 2.0 - 0.25 * LOCircuit.mutualInformation << std::endl << std::endl;

    outfile.close();

    if(2.0 - 0.25 * LOCircuit.mutualInformation > SUCCESS_MUT_ENT){

        outfile.open("Success.dat",std::ofstream::app);

        outfile << "H(X:Y): " << std::setprecision(16) << 2.0 - 0.25 * LOCircuit.mutualInformation << std::endl << std::endl;

        outfile << "U:\n" << std::setprecision(6) << U << std::endl << std::endl;

        for(int i=0;i<position.size();i++) outfile << std::setprecision(16) << position(i) << ",";

        outfile << std::endl << std::endl;

        outfile.close();

    }

    return;

}



Eigen::VectorXd MeritFunction::setInitialPosition(){

    U = Eigen::MatrixXcd::Identity(U.rows(),U.cols());

    Eigen::VectorXd position(funcDimension);

    int ampSize = U.rows() + ( U.rows() * U.rows() - U.rows() ) / 2;

    for(int j=0;j<INITIAL_CONDITION_RANDOM_DEGREE;j++){

        position = PI * Eigen::VectorXd::Random(funcDimension);

        for(int i=0;i<ampSize;i++) position(i) *= AMPLITUDE_SCALING;

        Eigen::MatrixXcd H( U.rows(),U.cols() );

        setAntiHermitian( H, position );

        Eigen::MatrixXcd UTemp( H.rows(),H.cols() );
        UTemp = H.exp();

        U = UTemp * U;

    }

    setPosition( U, position );

    return position;

}



void MeritFunction::setPosition(Eigen::MatrixXcd& U, Eigen::VectorXd& position){

    Eigen::MatrixXcd H(U.rows(),U.cols());

    std::complex<double> I(0.0,1.0);

    H = U.log();

    H /= I;

    int k = 0;

    for(int i=0;i<H.rows();i++) for(int j=i;j<H.cols();j++){

        position(k) = std::sqrt( std::norm(H(i,j)) ) ;
        if( i==j && std::real(H(i,j)) < 0 ) position(k) *= -1;
        k++;

    }

    for(int i=0;i<H.rows();i++) for(int j=i+1;j<H.cols();j++){

        position(k) = std::arg( H(i,j) );
        k++;

    }

    return;

}

void MeritFunction::setAntiHermitian( Eigen::MatrixXcd& H,Eigen::VectorXd& position ){

    int k = 0;

    for(int i=0;i<H.rows();i++) for(int j=i;j<H.cols();j++){

        H(i,j) = position(k);
        H(j,i) = position(k);

        k++;

    }

    std::complex<double> I(0.0,1.0);

    for(int i=0;i<H.rows();i++) for(int j=i+1;j<H.cols();j++){

        H(i,j) *= std::exp( I * position(k) );

        H(j,i) *= std::exp( -I * position(k) );

        k++;

    }

    H = I * H;

    return;

}


MeritFunction::MeritFunction(){



}


void MeritFunction::setToFullHilbertSpace(const int& subPhotons, const int& subModes,Eigen::MatrixXi& nv){

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
