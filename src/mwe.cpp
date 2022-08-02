#include <SYCL/sycl.hpp>
#include <cufft.h>
#include <iostream>

using namespace sycl;

int main(int argc, char *argv[])
{
    std::cout << "Starting mwe-hipsycl-cufft" << std::endl;

    sycl::queue myQueue;

    myQueue.submit([&](sycl::handler &cgh)
                   {
      // sycl::accessor access_cdata{buff_cdata, cgh, sycl::read_write};
      sycl::stream out(1024000, 102004, cgh);
      cgh.single_task([=]()
                       {
                        out << "I am a sycl single task\n";

                        __hipsycl_if_target_cuda(
                            out << ("I am a cuda test function\n");

                            // create FFT plan
                            cufftHandle fftPlan;
                            int n[] = {(int)4, (int)4, (int)4};
                            // cufftPlanMany(&fftPlan, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, 1);
                            // cufftPlan2d(&fftPlan, 4, 4, CUFFT_C2C);
                            cufftPlan3d(&fftPlan, 4, 4, 4, CUFFT_C2C);
                            // cufftComplex * data;

                            // for (int i = 0; i < dim; i++)
                            // {
                            //   for (int j = 0; j < dim; j++)
                            //   {
                            //     for (int k = 0; k < dim; k++)
                            //     {
                            //       int pos = (i * dim + j) * dim + k;
                            //       out << cdata[pos] << " ";
                            //     }
                            //     out << "\n";
                            //   }
                            //   out << "\n";
                            // }
                            // cudaMalloc((void **)&data, sizeof(cufftComplex) * dim * dim * dim);
                            // cudaMemcpy(data, cdata.data(), sizeof(cufftComplex) * dim * dim * dim, cudaMemcpyHostToDevice);
                            // cufftExecC2C(fftPlan, data, data, CUFFT_FORWARD);
                            // cudaMemcpy(cdata.data(), data, sizeof(cufftComplex) * dim * dim * dim, cudaMemcpyDeviceToHost);
                            // cudaDeviceSynchronize();
                            // for (int i = 0; i < dim; i++)
                            // {
                            //   for (int j = 0; j < dim; j++)
                            //   {
                            //     for (int k = 0; k < dim; k++)
                            //     {
                            //       int pos = (i * dim + j) * dim + k;
                            //       out << cdata[pos] << " ";
                            //     }
                            //     out << "\n";
                            //   }
                            //   out << "\n";
                            // }
                            // cufftDestroy(fftPlan);
                            // cudaFree(data);
                        );
                        }); });
    myQueue.wait();
}