# Builds the physics core into python/ (single precision, no FMA / fast-math: keep float behavior reproducible).
all: python/libspazsim.so
python/libspazsim.so: cpp/spazsim.cpp cpp/precalc_rands.h
	g++ -O2 -ffp-contract=off -shared -fPIC -std=c++17 -o $@ cpp/spazsim.cpp
tables:  # regenerate cpp/precalc_rands.h from a Ballistica checkout: make tables BA=/path/to/ballistica
	python3 cpp/gen_tables.py $(BA)/src/ballistica/shared/generic/utils.cc cpp/precalc_rands.h
test: all
	python3 tests/test_plugin_equiv.py brains/the_pad.npz
train: all
	cd python && python3 train.py --brain ../brains/the_pad.npz
clean:
	rm -f python/libspazsim.so
.PHONY: all tables test train clean
