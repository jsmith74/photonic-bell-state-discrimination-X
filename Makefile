CC = g++
CFLAGS = -Ofast -funroll-loops -c
LFLAGS = -Ofast -funroll-loops
OBJS = LinearOpticalTransform.o main.o GrayCode.o
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
	$(CC) $(CFLAGS) LinearOpticalTransform.cpp

clean:
	rm *.o LinearOpticalSimulation *.gcno *.gcda *.out *.txt
