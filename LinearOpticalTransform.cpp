/*! \file LinearOpticalTransform.cpp
 *  \brief This the general algorithm for simulating a linear optical quantum circuit.
 *
 *     For more information, refer to:
 *      https://arxiv.org/abs/1711.01319
 *
 *
 *
 * \author Jake Smith <jsmith74@tulane.edu>
 */


#include "LinearOpticalTransform.h"

#define MODES 12

LinearOpticalTransform::LinearOpticalTransform(){


}

void LinearOpticalTransform::initializeCircuit(Eigen::MatrixXi& inBasis, Eigen::MatrixXi& outBasis){

    n.resize( inBasis.rows() );
    m.resize( inBasis.rows() );

    nPrime.resize( outBasis.rows() );
    mPrime.resize( outBasis.rows() );

    photons = inBasis.row(0).sum();

    assert( photons == 8 );

    for(int i=0;i<inBasis.rows();i++){

        assert( inBasis.row(i).sum() == photons && "Error: Photon number must be preserved you have included some input basis states that do not have the correct number of photons." );

        n.at(i).resize( inBasis.cols() );

        for(int j=0;j<inBasis.cols();j++) n.at(i).at(j) = inBasis(i,j);

        m.at(i).resize( photons );

        setmVec( m.at(i), n.at(i) );

    }

    for(int i=0;i<outBasis.rows();i++){

        assert( outBasis.row(i).sum() == photons && "Error: Photon number must be preserved you have included some output basis states that do not have the correct number of photons." );

        nPrime.at(i).resize( outBasis.cols() );

        for(int j=0;j<outBasis.cols();j++) nPrime.at(i).at(j) = outBasis(i,j);

        mPrime.at(i).resize( photons );

        setmVec( mPrime.at(i), nPrime.at(i) );

    }

    factorial.resize( photons + 1 );

    for(int i=0;i<factorial.size();i++) factorial[i] = doublefactorial(i);

    useRysers.resize( outBasis.rows() );

    for(int i=0;i<outBasis.rows();i++){

        if( std::pow(2.0,photons) < numbPermutations(i) ) useRysers[i] = true;
        else useRysers[i] = false;

    }

    inBasis.resize(0,0);
    outBasis.resize(0,0);

    graycode.initialize( photons );

    bellStates = Eigen::Matrix4d::Zero();

    bellStates(0,0) = 1/sqrt(2.0);
    bellStates(1,0) = 1/sqrt(2.0);

    bellStates(0,1) = 1/sqrt(2.0);
    bellStates(1,1) = -1/sqrt(2.0);

    bellStates(2,2) = 1/sqrt(2.0);
    bellStates(3,2) = 1/sqrt(2.0);

    bellStates(2,3) = 1/sqrt(2.0);
    bellStates(3,3) = -1/sqrt(2.0);

    return;

}

void LinearOpticalTransform::setMutualInformation(Eigen::MatrixXcd& U){

    mutualInformation = 0;

    for(int y=0;y<useRysers.size();y++){

        if( useRysers[y] == true ) rysersAlgorithm(U,y);

        else permutationAlgorithm(U,y);

    }

    return;

}

inline int idx( int& i,int& j ){

    return 2 * (i + MODES * j);

}

inline void spec_complex_mult_function(double* z1,double* z2){

    double rp = z1[0] * z2[0] - z1[1] * z2[1];
    z1[1] = z1[0] * z2[1] + z1[1] * z2[0];
    z1[0] = rp;

    return;

}

void LinearOpticalTransform::rysersAlgorithm(Eigen::MatrixXcd& U,int& i){

    double* dev_U = (double*)U.data();

    double bosonOutput = 1.0;

    for(int p=0;p<nPrime[i].size();p++) bosonOutput *= factorial[ nPrime[i][p] ];

    Eigen::Vector4cd A = Eigen::Vector4cd::Zero();

    for(int j=0;j<4;j++){

        //#pragma omp simd
        //for(int l=0;l<16;l++) weights[l] = 0;

        bool even = true;

        Eigen::ArrayXcd weights = Eigen::ArrayXcd::Zero( 8 );

        while( graycode.iterate() ){

            weights(0) += boolPow( graycode.sign ) * U.coeffRef( m[j][0],mPrime[i][graycode.j] );
            weights(1) += boolPow( graycode.sign ) * U.coeffRef( m[j][1],mPrime[i][graycode.j] );
            weights(2) += boolPow( graycode.sign ) * U.coeffRef( m[j][2],mPrime[i][graycode.j] );
            weights(3) += boolPow( graycode.sign ) * U.coeffRef( m[j][3],mPrime[i][graycode.j] );
            weights(4) += boolPow( graycode.sign ) * U.coeffRef( m[j][4],mPrime[i][graycode.j] );
            weights(5) += boolPow( graycode.sign ) * U.coeffRef( m[j][5],mPrime[i][graycode.j] );
            weights(6) += boolPow( graycode.sign ) * U.coeffRef( m[j][6],mPrime[i][graycode.j] );
            weights(7) += boolPow( graycode.sign ) * U.coeffRef( m[j][7],mPrime[i][graycode.j] );

            A(j) -= boolPow( even ) * weights(0) * weights(1) * weights(2) * weights(3) * weights(4) * weights(5) * weights(6) * weights(7);

            even = !even;

}

        double bosonInput = 1.0;

        for(int p=0;p<n[j].size();p++) bosonInput *= factorial[ n[j][p] ];

        A.coeffRef(j) /= sqrt( bosonInput * bosonOutput );

    }

    Eigen::MatrixXcd stateAmplitude = A.transpose() * bellStates;

    double py[4];

    py[0] = std::norm( stateAmplitude.coeffRef(0,0) );
    py[1] = std::norm( stateAmplitude.coeffRef(0,1) );
    py[2] = std::norm( stateAmplitude.coeffRef(0,2) );
    py[3] = std::norm( stateAmplitude.coeffRef(0,3) );

    double pytotal = py[0] + py[1] + py[2] + py[3];

    if(py[0] != 0) mutualInformation += py[0] * log2( pytotal / py[0] );
    if(py[1] != 0) mutualInformation += py[1] * log2( pytotal / py[1] );
    if(py[2] != 0) mutualInformation += py[2] * log2( pytotal / py[2] );
    if(py[3] != 0) mutualInformation += py[3] * log2( pytotal / py[3] );

    return;

}

void LinearOpticalTransform::permutationAlgorithm(Eigen::MatrixXcd& U,int& i){

    Eigen::Vector4cd A = Eigen::Vector4cd::Zero();

    do{

        for(int j=0;j<4;j++){

            std::complex<double> Uprod(1.0,0.0);

            for(int k=0;k<m[j].size();k++){

                Uprod *= U.coeffRef( m[j][k],mPrime[i][k] );

            }

            A.coeffRef(j) += Uprod;

        }

    } while( std::next_permutation( mPrime[i].begin(), mPrime[i].end() ) );

    double bosonNum = 1.0;

    for(int p=0;p<U.rows();p++) bosonNum *= factorial[ nPrime[i][p] ];

    for(int j=0;j<4;j++){

        double bosonDen = 1.0;

        for(int p=0;p<U.rows();p++) bosonDen *= factorial[ n[j][p] ];

        A.coeffRef(j) *= sqrt( bosonNum/bosonDen );

    }

    Eigen::MatrixXcd stateAmplitude = A.transpose() * bellStates;

    double py[4];

    py[0] = std::norm( stateAmplitude.coeffRef(0,0) );
    py[1] = std::norm( stateAmplitude.coeffRef(0,1) );
    py[2] = std::norm( stateAmplitude.coeffRef(0,2) );
    py[3] = std::norm( stateAmplitude.coeffRef(0,3) );

    double pytotal = py[0] + py[1] + py[2] + py[3];

    if(py[0] != 0) mutualInformation += py[0] * log2( pytotal / py[0] );
    if(py[1] != 0) mutualInformation += py[1] * log2( pytotal / py[1] );
    if(py[2] != 0) mutualInformation += py[2] * log2( pytotal / py[2] );
    if(py[3] != 0) mutualInformation += py[3] * log2( pytotal / py[3] );

    return;

}

double LinearOpticalTransform::numbPermutations(int& i){

    double output = factorial[photons];

    for(int j=0;j<nPrime[i].size();j++) output /= factorial[ nPrime[i][j] ];

    return output;

}

void LinearOpticalTransform::setmVec(std::vector<int>& m, std::vector<int>& n){

    int k=0;

    for(int i=0;i<n.size();i++){

        for(int j=0;j<n.at(i);j++){

            m.at(k) = i;

            k++;

        }

    }

    return;
}

template <typename T>
void LinearOpticalTransform::printVec(std::vector<T>& a){

    for(int i=0;i<a.size();i++) std::cout << a[i] << " ";

    std::cout << std::endl;

    return;

}

double LinearOpticalTransform::doublefactorial(int x){

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
