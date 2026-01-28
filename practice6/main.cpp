// Practical Work 6
// OpenCL: CPU vs GPU
// Элементное сложение векторов и матричное умножение

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>

#define N 1024

std::string loadKernel(const char* name) {
    std::ifstream file(name);
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
}

int main() {
    std::vector<float> A(N, 1.0f), B(N, 2.0f), C(N);

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    queue = clCreateCommandQueue(context, device, 0, nullptr);

    std::string source = loadKernel("vector_add.cl");
    const char* src = source.c_str();

    cl_program program = clCreateProgramWithSource(context, 1, &src, nullptr, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, "vector_add", nullptr);

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * N, A.data(), nullptr);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * N, B.data(), nullptr);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                 sizeof(float) * N, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);

    size_t globalSize = N;

    clock_t start = clock();
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                           &globalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    clock_t end = clock();

    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                         sizeof(float) * N, C.data(), 0, nullptr, nullptr);

    std::cout << "GPU time: "
              << double(end - start) / CLOCKS_PER_SEC
              << " sec\n";

    // CPU version
    start = clock();
    for (int i = 0; i < N; i++)
        C[i] = A[i] + B[i];
    end = clock();

    std::cout << "CPU time: "
              << double(end - start) / CLOCKS_PER_SEC
              << " sec\n";

    return 0;
}
