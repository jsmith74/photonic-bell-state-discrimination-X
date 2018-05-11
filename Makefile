CC = g++
CFLAGS = -Ofast -funroll-loops -c
LFLAGS = -Ofast -funroll-loops
OBJS = LinearOpticalTransform.o main.o GrayCode.o MeritFunction.o BFGS_Optimization.o
OMPFLAGS = -fopenmp
PFLAGS = -pg -g -fprofile-arcs -ftest-coverage

all: LinearOpticalSimulation

LinearOpticalSimulation: $(OBJS)
	$(CC) $(LFLAGS) $(OMPFLAGS) $(OBJS) -o LinearOpticalSimulation

GrayCode.o: GrayCode.cpp
	$(CC) $(CFLAGS) GrayCode.cpp

main.o: main.cpp
	$(CC) $(CFLAGS) $(OMPFLAGS) main.cpp

LinearOpticalTransform.o: LinearOpticalTransform.cpp
	$(CC) $(CFLAGS) $(OMPFLAGS) LinearOpticalTransform.cpp

MeritFunction.o: MeritFunction.cpp
	$(CC) $(CFLAGS) MeritFunction.cpp

BFGS_Optimization.o: BFGS_Optimization.cpp
	$(CC) $(CFLAGS) BFGS_Optimization.cpp

clean:
	rm *.o LinearOpticalSimulation *.gcno *.gcda *.out *.txt
