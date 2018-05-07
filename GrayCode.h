#ifndef GRAYCODE_H_INCLUDED
#define GRAYCODE_H_INCLUDED

#include <vector>
#include <iostream>
#include <assert.h>

class GrayCode{

    public:

        GrayCode();
        void initialize(int size);
        bool iterate();
        void printBitstring();
        bool sign;
        int j;

    private:

        int n;
        int k;
        bool ending;

        std::vector<bool> bitstring;

        template <typename T>
        void printVec(std::vector<T>& a);

};

#endif // GRAYCODE_H_INCLUDED
