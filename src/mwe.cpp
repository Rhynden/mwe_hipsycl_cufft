#include <SYCL/sycl.hpp>
#include <hipfft/hipfft.h>
#include <iostream>
#include <math.h>
#include <complex>
#include <vector>

using namespace sycl;

int main(int argc, char *argv[])
{
    std::cout << "hipfft 3D double-precision complex-to-complex transform\n";

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
    size_t complex_bytes = sizeof(decltype(cdata)::value_type) * cdata.size();
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
            hipError_t hip_rt;
            hipfftComplex *x;
            hip_rt = hipMalloc(&x, complex_bytes);
            if (hip_rt != hipSuccess)
                throw std::runtime_error("hipMalloc failed");

            hip_rt = hipMemcpy(x, cdata.data(), complex_bytes, hipMemcpyHostToDevice);
            if (hip_rt != hipSuccess)
                throw std::runtime_error("hipMemcpy failed");

            // create FFT plan
            // Create plan
            hipfftHandle plan;
            hipfftResult hipfft_rt = hipfftCreate(&plan);
            if (hipfft_rt != HIPFFT_SUCCESS)
                throw std::runtime_error("failed to create plan");

            hipfft_rt = hipfftPlan3d(&plan,       // plan handle
                                     4,          // transform length
                                     4,          // transform length
                                     4,          // transform length
                                     HIPFFT_C2C); // transform type (HIPFFT_C2C for single-precision)

            if (hipfft_rt != HIPFFT_SUCCESS)
                throw std::runtime_error("hipfftPlan3d failed");

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

            // Execute plan
            // hipfftExecZ2Z: double precision, hipfftExecC2C: for single-precision
            hipfft_rt = hipfftExecC2C(plan, x, x, HIPFFT_FORWARD);
            if (hipfft_rt != HIPFFT_SUCCESS)
                throw std::runtime_error("hipfftExecZ2Z failed");

            hip_rt = hipMemcpy((void*)cdata.data(), x, complex_bytes, hipMemcpyDeviceToHost);
            if (hip_rt != hipSuccess)
                throw std::runtime_error("hipMemcpy failed");

            out << "After hipfft\n";
            // for (int i = 0; i < dim; i++)
            // {
            //     for (int j = 0; j < dim; j++)
            //     {
            //         for (int k = 0; k < dim; k++)
            //         {
            //             int pos = (i * dim + j) * dim + k;
            //             out << cdata[pos].real() << " " << cdata[pos].imag() << " ";
            //         }
            //         out << "\n";
            //     }
            //     out << "\n";
            // }
            
            // hipfftDestroy(plan);
            // hipFree(x); 
    }); });
    // cufftDestroy(fftPlan);
    // cudaFree(data);}); });

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