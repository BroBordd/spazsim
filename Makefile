# Physics core. Single precision, no FMA contraction, no fast-math: float results stay bit-reproducible.
# OpenMP (multithreaded env pool) is used when the compiler supports it; otherwise it builds single-threaded.
OMP := $(shell echo 'int main(){}' | g++ -fopenmp -x c++ - -o /dev/null 2>/dev/null && echo -fopenmp)
CXXFLAGS := -O3 -ffp-contract=off $(OMP) -shared -fPIC -std=c++17

all: python/libspazsim.so
python/libspazsim.so: cpp/spazsim.cpp cpp/precalc_rands.h
	g++ $(CXXFLAGS) -o $@ cpp/spazsim.cpp
tables:  # regenerate cpp/precalc_rands.h from a Ballistica checkout: make tables BA=/path/to/ballistica
	python3 cpp/gen_tables.py $(BA)/src/ballistica/shared/generic/utils.cc cpp/precalc_rands.h
test: all
	python3 tests/test_plugin_equiv.py brains/the_pad.npz
	python3 tests/test_pool_equiv.py
	python3 tests/test_grad_shards.py
hash: all  # regression guard for the physics core on THIS toolchain (libm may differ elsewhere): prints sha256 of a scripted trajectory
	python3 tests/trajectory_hash.py
train: all
	cd python && python3 train.py
clean:
	rm -f python/libspazsim.so
.PHONY: all tables test hash train clean
