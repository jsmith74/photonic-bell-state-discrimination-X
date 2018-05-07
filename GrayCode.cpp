#include "GrayCode.h"


GrayCode::GrayCode(){


}

void GrayCode::initialize(int size){

    bitstring.resize(size);

    for(int i=0;i<size;i++) bitstring[i] = 0;

    j = 0;

    k = 0;

    n = size;

    ending = false;

    return;

}

bool GrayCode::iterate(){

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

void GrayCode::printBitstring(){

    printVec( bitstring );

    return;

}

template <typename T>
void GrayCode::printVec(std::vector<T>& a){

    for(int i=0;i<a.size();i++) std::cout << a[i] << " ";

    std::cout << std::endl;

    return;

}
