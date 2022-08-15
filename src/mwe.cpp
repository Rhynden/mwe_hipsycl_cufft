#include <SYCL/sycl.hpp>
#include <cufft.h>
#include <iostream>
#include <math.h>
#include <complex>
#include <vector>

using namespace sycl;

int main(int argc, char *argv[])
{
    std::cout << "Starting mwe-hipsycl-cufft" << std::endl;

    sycl::queue myQueue;

    // sycl::buffer<float, 3> some_buff{sycl::range<3>{64, 64, 64}};

    // myQueue.submit([&](handler &cgh)
    //                {
    // The kernel writes a, so get a write accessor on it
    // accessor A{some_buff, cgh, write_only};

    // // Enqueue a parallel kernel iterating on a N*M 2D iteration space
    // cgh.parallel_for(range<3> {64,64,64}, [=](id<3> index) {
    //   A[index] = index[0] * 4 + index[1] *2 + index[0];
    // }); });

    std::vector<std::complex<float>> cdata(64);
    const unsigned int dim = 4;
    for (size_t i = 0; i < dim * dim * dim; i++)
    {
        cdata[i] = i;
        cdata[i].imag(1.0f);
    }

    myQueue.submit([&](sycl::handler &cgh)
                   {
        // sycl::accessor access_cdata{buff_cdata, cgh, sycl::read_write};
        sycl::stream out(1024000, 102004, cgh);

        // auto acc = some_buff.get_access<sycl::access::mode::read>(cgh);

        cgh.hipSYCL_enqueue_custom_operation([=](sycl::interop_handle &h)
                                             {
            out << "I am a sycl custom task\n";
            // Can extract device pointers from accessors
            // void *native_mem = h.get_native_mem<sycl::backend::hip>(acc);
            // Can extract stream (note: get_native_queue() may not be
            // supported on CPU backends)
            // hipStream_t stream = h.get_native_queue<sycl::backend::hip>();
            // Can extract HIP device (note: get_native_device() may not be
            // supported on CPU backends)
            // int dev = h.get_native_device<sycl::backend::hip>();
            // Can enqueue arbitrary backend operations. This could also be a kernel launch
            // or a call to a library that enqueues operations on the stream etc

            // create FFT plan
            cufftHandle fftPlan;
            // int n[] = {(int)4, (int)4, (int)4};
            // cufftPlanMany(&fftPlan, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, 1);
            // cufftPlan2d(&fftPlan, 4, 4, CUFFT_C2C);
            cufftPlan3d(&fftPlan, dim, dim, dim, CUFFT_C2C);
            // cufftPlan1d(&fftPlan, dim* dim* dim, CUFFT_C2C, 1);
            cufftComplex *data;

            for (int i = 0; i < dim; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    for (int k = 0; k < dim; k++)
                    {
                        int pos = (i * dim + j) * dim + k;
                        out << cdata[pos].real() << " " << cdata[pos].imag() << " ";
                    }
                    out << "\n";
                }
                out << "\n";
            }
            cudaMalloc((void **)&data, sizeof(cufftComplex) * dim * dim * dim);
            cudaMemcpy(data, cdata.data(), sizeof(cufftComplex) * dim * dim * dim, cudaMemcpyHostToDevice);
            cufftExecC2C(fftPlan, data, data, CUFFT_FORWARD);
            cudaMemcpy((void *)cdata.data(), data, sizeof(cufftComplex) * dim * dim * dim, cudaMemcpyDeviceToHost);
            // cudaDeviceSynchronize();
            out << "After cufft\n";
            for (int i = 0; i < dim; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    for (int k = 0; k < dim; k++)
                    {
                        int pos = (i * dim + j) * dim + k;
                        out << cdata[pos].real() << " " << cdata[pos].imag() << " ";
                    }
                    out << "\n";
                }
                out << "\n";
            }
            cufftDestroy(fftPlan);
            cudaFree(data);}); });

    //   hipMemcpyAsync(target_ptr, native_mem, test_size * sizeof(int),
    //                   hipMemcpyDeviceToHost, stream); });
    // cgh.single_task([=]()
    //                       {
    //                     out << "I am a sycl single task\n";

    //                     __hipsycl_if_target_cuda(
    //                         out << ("I am a cuda test function\n");

    //                         // create FFT plan
    //                         cufftHandle fftPlan;
    //                         int n[] = {(int)4, (int)4, (int)4};
    //                         // cufftPlanMany(&fftPlan, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, 1);
    //                         // cufftPlan2d(&fftPlan, 4, 4, CUFFT_C2C);
    //                         cufftPlan3d(&fftPlan, 4, 4, 4, CUFFT_C2C);
    //                         // cufftComplex * data;

    //                         // for (int i = 0; i < dim; i++)
    //                         // {
    //                         //   for (int j = 0; j < dim; j++)
    //                         //   {
    //                         //     for (int k = 0; k < dim; k++)
    //                         //     {
    //                         //       int pos = (i * dim + j) * dim + k;
    //                         //       out << cdata[pos] << " ";
    //                         //     }
    //                         //     out << "\n";
    //                         //   }
    //                         //   out << "\n";
    //                         // }
    //                         // cudaMalloc((void **)&data, sizeof(cufftComplex) * dim * dim * dim);
    //                         // cudaMemcpy(data, cdata.data(), sizeof(cufftComplex) * dim * dim * dim, cudaMemcpyHostToDevice);
    //                         // cufftExecC2C(fftPlan, data, data, CUFFT_FORWARD);
    //                         // cudaMemcpy(cdata.data(), data, sizeof(cufftComplex) * dim * dim * dim, cudaMemcpyDeviceToHost);
    //                         // cudaDeviceSynchronize();
    //                         // for (int i = 0; i < dim; i++)
    //                         // {
    //                         //   for (int j = 0; j < dim; j++)
    //                         //   {
    //                         //     for (int k = 0; k < dim; k++)
    //                         //     {
    //                         //       int pos = (i * dim + j) * dim + k;
    //                         //       out << cdata[pos] << " ";
    //                         //     }
    //                         //     out << "\n";
    //                         //   }
    //                         //   out << "\n";
    //                         // }
    //                         // cufftDestroy(fftPlan);
    //                         // cudaFree(data);
    //                     ); }); });
    myQueue.wait();
}