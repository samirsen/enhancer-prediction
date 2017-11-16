process: process.cc
	g++ -O3 -std=c++11 process.cc -o process

clean:
	rm process
