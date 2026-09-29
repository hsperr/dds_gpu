# CPU tools build with any C++17 compiler; the GPU driver needs nvcc.
#   make            CPU solver + test and bench tools into build/
#   make check      run tests/check.sh
#   make gpu SM=120 CUDA batch + streaming drivers (SM = compute capability, e.g. 89, 90, 120)
CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -march=native -pthread
SM ?= 120
HDR := $(wildcard src/*.h)
B := build

all: $(B)/dd_cpu $(B)/test_small $(B)/cmp_tt $(B)/wave_test $(B)/bench_k $(B)/pbn $(B)/brute_one $(B)/verify_one $(B)/dd_pbn

$(B):
	mkdir -p $(B)

$(B)/dd_cpu: cpu/cpu_main.cpp $(HDR) | $(B)
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $<
$(B)/bench_k: bench/bench_k.cpp $(HDR) | $(B)
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $<
$(B)/verify_one: tests/verify_one.cpp $(HDR) | $(B)
	$(CXX) $(CXXFLAGS) -Isrc -DDD_VERIFY -o $@ $<
$(B)/%: tests/%.cpp $(HDR) | $(B)
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $<

pbn: $(B)/pbn

gpu: $(B)/dd_gpu $(B)/dd_stream
$(B)/dd_gpu: gpu/gpu_main.cu $(HDR) | $(B)
	nvcc -O3 -std=c++17 -arch=sm_$(SM) -Isrc -o $@ $<
$(B)/dd_stream: gpu/stream_main.cu $(HDR) | $(B)
	nvcc -O3 -std=c++17 -arch=sm_$(SM) -Isrc -o $@ $<

check:
	tests/check.sh

clean:
	rm -rf $(B)

.PHONY: all pbn gpu check clean
