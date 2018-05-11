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
#define MIC_RATIO 0


int* dev_parallelGrid;
int dev_numThreadsCPU;

LinearOpticalTransform::LinearOpticalTransform(){


}

void LinearOpticalTransform::setParallelGrid(int micRatio){

    checkThreadsAndProcs();

    std::cout << "CPU threads: " << numThreadsCPU << std::endl;
    std::cout << "# coprocessors: " << numCoprocessors << std::endl;
    std::cout << "# coprocessor threads: " << numThreadsPhi << std::endl;

    setTotalTerms();

    coprocessorTerms = ( micRatio / 100.0 ) * totalTerms;
    CPUTerms = totalTerms - coprocessorTerms;

    CPUTermsPerThread = CPUTerms / numThreadsCPU;

    allocateWorkToThreads();

    printParallelGrid();

    return;

}

void LinearOpticalTransform::printParallelGrid(){

    std::ofstream outfile("parallelGrid.dat");

    for(int i=0;i<numThreadsPhi + numThreadsCPU;i++){

        int terms = 0;

        for(int y=dev_parallelGrid[i];y<dev_parallelGrid[i+1];y++){

            if( useRysers[y] ) terms += std::pow(2,photons);
            else terms += numbPermutations(y);

        }

        outfile << i << "\t" << terms << std::endl;

    }

    outfile.close();

    return;

}

void LinearOpticalTransform::allocateWorkToThreads(){

    dev_parallelGrid = new int[numThreadsPhi + numThreadsCPU + 1];
    dev_parallelGrid[0] = 0;

    int localTerms = 0;

    int k=1;

    int savePoint;

    for(int i=0;i<useRysers.size() && k!=numThreadsCPU+1 ;i++){

        if( useRysers[i] ) localTerms += std::pow(2,photons);
        else localTerms += numbPermutations(i);

        if( localTerms >= CPUTermsPerThread ){

            dev_parallelGrid[k] = i;
            localTerms = 0;
            k++;

        }

        savePoint = i+1;

    }

    for(int i=savePoint;i<useRysers.size() && k!=numThreadsPhi + numThreadsCPU + 1;i++){

        if( useRysers[i] ) localTerms += std::pow(2,photons);
        else localTerms += numbPermutations(i);

        if( localTerms >= coprocessorTermsPerThread ){

            dev_parallelGrid[k] = i;
            localTerms = 0;
            k++;

        }

        savePoint = i+1;

    }

    assert( savePoint == useRysers.size() );

    dev_parallelGrid[ k ] = useRysers.size();

    k++;

    for(;k<numThreadsPhi + numThreadsCPU + 1;k++){

        dev_parallelGrid[ k ] = 0;

    }

    return;

}

void LinearOpticalTransform::setTotalTerms(){

    totalTerms = 0;

    for(int i=0;i<useRysers.size();i++){

        if( useRysers[i] ) totalTerms += std::pow(2,photons);
        else totalTerms += numbPermutations(i);

    }

    return;

}

void LinearOpticalTransform::checkThreadsAndProcs(){

#pragma omp parallel
{

    numThreadsCPU = omp_get_num_threads();

}

    numCoprocessors = 0;//_Offload_number_of_devices();

//#pragma offload target (mic:0) inout(numThreadsPhi)
//#pragma omp parallel
//{
//
//    int threadID = omp_get_thread_num();
//
//    if(threadID == 0) numThreadsPhi = omp_get_num_threads();
//
//}

    numThreadsPhi = 0;

    numThreadsPhi *= numCoprocessors;

    assert( numCoprocessors == 0 );

    //dev_numThreadsCPU = numThreadsCPU;

//#pragma offload target(mic:0) in( dev_numThreadsCPU : ALLOC RETAIN )
//#pragma omp parallel
//{
//
//}
//
//#pragma offload target(mic:1) in( dev_numThreadsCPU : ALLOC RETAIN )
//#pragma omp parallel
//{
//
//}

    return;

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

    setParallelGrid(MIC_RATIO);

    return;

}

void LinearOpticalTransform::setMutualInformation(Eigen::MatrixXcd& U){

    double parallelMutualInformation = 0;

#pragma omp parallel reduction(+:parallelMutualInformation)
    {

        int threadID = omp_get_thread_num();

        for( int y=dev_parallelGrid[threadID]; y<dev_parallelGrid[threadID+1]; y++ ){

            if( useRysers[y] ) rysersAlgorithm(U,y,parallelMutualInformation);

            else permutationAlgorithm(U,y,parallelMutualInformation);

        }

    }

    mutualInformation = parallelMutualInformation;

    return;

}

inline int idx( int& i,int& j ){

    return 2 * (i + MODES * j);

}

inline void spec_complex_mult_function(double& x1,double& y1,double& x2,double& y2){

    double rp = x1 * x2 - y1 * y2;
    y1 = x1 * y2 + x2 * y1;
    x1 = rp;

    return;

}

bool iterate(bool& sign,int& j, int& k,std::vector<bool>& bitstring,bool& ending,int& n ){

    if( ending ){

        bitstring[n-1] = 0;
        k = 0;
        ending = false;
        return false;

    }

    bool t = k % 2;     j = 0;

    if( t == 1 ){

        while( bitstring[j] != 1 ) j++;
        j++;

    }

    bitstring[j] = 1 - bitstring[j];
    sign = bitstring[j];

    k += 2 * bitstring[j] - 1;

    if( k == bitstring[n-1] ) ending = true;

    return true;

}

void LinearOpticalTransform::rysersAlgorithm(Eigen::MatrixXcd& U,int& i,double& parallelMutualInformation){

    bool sign;
    int graycodej, graycodek;
    std::vector<bool> bitstring(photons);
    for(int p=0;p<photons;p++) bitstring[p] = 0;

    graycodej = 0;

    graycodek = 0;

    bool ending = false;

    double* dev_U = (double*)U.data();

    double bosonOutput = 1.0;

    for(int p=0;p<nPrime[i].size();p++) bosonOutput *= factorial[ nPrime[i][p] ];

    double A[8];

    A[0] = 0;   A[1] = 0;
    A[2] = 0;   A[3] = 0;
    A[4] = 0;   A[5] = 0;
    A[6] = 0;   A[7] = 0;

    for(int j=0;j<4;j++){

        double dev_weights[16];

        #pragma omp simd
        for(int l=0;l<16;l++) dev_weights[l] = 0;

        bool even = true;

        while( iterate(sign,graycodej,graycodek,bitstring,ending,photons) ){

            if( sign ){

                #pragma omp simd
                for(int l=0;l<8;l++){

                    dev_weights[ 2*l ] += dev_U[ idx( m[j][l],mPrime[i][graycodej] ) ];
                    dev_weights[ 2*l + 1 ] += dev_U[ idx( m[j][l],mPrime[i][graycodej] ) + 1 ];

                }

            }

            else{

                #pragma omp simd
                for(int l=0;l<8;l++){

                    dev_weights[ 2*l ] -= dev_U[ idx( m[j][l],mPrime[i][graycodej] ) ];
                    dev_weights[ 2*l + 1 ] -= dev_U[ idx( m[j][l],mPrime[i][graycodej] ) + 1 ];

                }

            }

            double resX = 1;    double resY = 0;

            spec_complex_mult_function(resX,resY,dev_weights[0],dev_weights[1]);
            spec_complex_mult_function(resX,resY,dev_weights[2],dev_weights[3]);
            spec_complex_mult_function(resX,resY,dev_weights[4],dev_weights[5]);
            spec_complex_mult_function(resX,resY,dev_weights[6],dev_weights[7]);
            spec_complex_mult_function(resX,resY,dev_weights[8],dev_weights[9]);
            spec_complex_mult_function(resX,resY,dev_weights[10],dev_weights[11]);
            spec_complex_mult_function(resX,resY,dev_weights[12],dev_weights[13]);
            spec_complex_mult_function(resX,resY,dev_weights[14],dev_weights[15]);

            A[2*j]   -= boolPow( even ) * resX;
            A[2*j+1] -= boolPow( even ) * resY;

            even = !even;

        }

        double bosonInput = 1.0;

        for(int p=0;p<n[j].size();p++) bosonInput *= factorial[ n[j][p] ];

        A[2*j]   /= sqrt( bosonInput * bosonOutput );
        A[2*j+1] /= sqrt( bosonInput * bosonOutput );

    }

    double stateAmplitude[8];

    stateAmplitude[0] =  0.7071067811865475 * (A[0] + A[2]);
    stateAmplitude[1] =  0.7071067811865475 * (A[1] + A[3]);

    stateAmplitude[2] =  0.7071067811865475 * (A[0] - A[2]);
    stateAmplitude[3] =  0.7071067811865475 * (A[1] - A[3]);

    stateAmplitude[4] =  0.7071067811865475 * (A[4] + A[6]);
    stateAmplitude[5] =  0.7071067811865475 * (A[5] + A[7]);

    stateAmplitude[6] =  0.7071067811865475 * (A[4] - A[6]);
    stateAmplitude[7] =  0.7071067811865475 * (A[5] - A[7]);

    double py[4];

    py[0] = stateAmplitude[0] * stateAmplitude[0] + stateAmplitude[1] * stateAmplitude[1];
    py[1] = stateAmplitude[2] * stateAmplitude[2] + stateAmplitude[3] * stateAmplitude[3];
    py[2] = stateAmplitude[4] * stateAmplitude[4] + stateAmplitude[5] * stateAmplitude[5];
    py[3] = stateAmplitude[6] * stateAmplitude[6] + stateAmplitude[7] * stateAmplitude[7];

    double pytotal = py[0] + py[1] + py[2] + py[3];

    if(py[0] != 0) parallelMutualInformation += py[0] * log2( pytotal / py[0] );
    if(py[1] != 0) parallelMutualInformation += py[1] * log2( pytotal / py[1] );
    if(py[2] != 0) parallelMutualInformation += py[2] * log2( pytotal / py[2] );
    if(py[3] != 0) parallelMutualInformation += py[3] * log2( pytotal / py[3] );

    return;

}

void LinearOpticalTransform::permutationAlgorithm(Eigen::MatrixXcd& U,int& i,double& parallelMutualInformation){

    double* dev_U = (double*)U.data();

    double A[8];

    A[0] = 0;   A[1] = 0;
    A[2] = 0;   A[3] = 0;
    A[4] = 0;   A[5] = 0;
    A[6] = 0;   A[7] = 0;

    do{

        for(int j=0;j<4;j++){

            double resX = 1;    double resY = 0;

            for(int k=0;k<m[j].size();k++){

                spec_complex_mult_function( resX,resY,dev_U[ idx(m[j][k],mPrime[i][k]) ], dev_U[ idx(m[j][k],mPrime[i][k]) + 1 ] );

            }

            A[2*j]   += resX;
            A[2*j+1] += resY;

        }

    } while( std::next_permutation( mPrime[i].begin(), mPrime[i].end() ) );

    double bosonNum = 1.0;

    for(int p=0;p<U.rows();p++) bosonNum *= factorial[ nPrime[i][p] ];

    for(int j=0;j<4;j++){

        double bosonDen = 1.0;

        for(int p=0;p<U.rows();p++) bosonDen *= factorial[ n[j][p] ];

        A[2*j]   *= sqrt( bosonNum/bosonDen );
        A[2*j+1] *= sqrt( bosonNum/bosonDen );

    }

    double stateAmplitude[8];

    stateAmplitude[0] =  0.7071067811865475 * (A[0] + A[2]);
    stateAmplitude[1] =  0.7071067811865475 * (A[1] + A[3]);

    stateAmplitude[2] =  0.7071067811865475 * (A[0] - A[2]);
    stateAmplitude[3] =  0.7071067811865475 * (A[1] - A[3]);

    stateAmplitude[4] =  0.7071067811865475 * (A[4] + A[6]);
    stateAmplitude[5] =  0.7071067811865475 * (A[5] + A[7]);

    stateAmplitude[6] =  0.7071067811865475 * (A[4] - A[6]);
    stateAmplitude[7] =  0.7071067811865475 * (A[5] - A[7]);

    double py[4];

    py[0] = stateAmplitude[0] * stateAmplitude[0] + stateAmplitude[1] * stateAmplitude[1];
    py[1] = stateAmplitude[2] * stateAmplitude[2] + stateAmplitude[3] * stateAmplitude[3];
    py[2] = stateAmplitude[4] * stateAmplitude[4] + stateAmplitude[5] * stateAmplitude[5];
    py[3] = stateAmplitude[6] * stateAmplitude[6] + stateAmplitude[7] * stateAmplitude[7];

    double pytotal = py[0] + py[1] + py[2] + py[3];

    if(py[0] != 0) parallelMutualInformation += py[0] * log2( pytotal / py[0] );
    if(py[1] != 0) parallelMutualInformation += py[1] * log2( pytotal / py[1] );
    if(py[2] != 0) parallelMutualInformation += py[2] * log2( pytotal / py[2] );
    if(py[3] != 0) parallelMutualInformation += py[3] * log2( pytotal / py[3] );

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
