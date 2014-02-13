all: release

debug: gpu_no_opt_nn
	nvcc -g -o gpu_no_nn_nn gpu_no_opt_nn.cu

release: gpu_no_opt_nn.cu
	nvcc -o gpu_no_opt_nn gpu_no_opt_nn.cu

clean:
	rm gpu_no_opt_nn
