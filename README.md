## Solve Poisson equation using CUDA FFT

When using Visual studio, click right button on your project name. Then click on properties. 

Then configuration properties, linker, input. In additional dependencies you must write cufft.lib.

## Poisson equation
 ![image](https://github.com/wangjinlong9788/Solve-Poisson-equation-using-CUDA-FFT/blob/master/Possion.PNG)

## In Mathematical Methods of Physics, a Poissone quation can be solved using Fourier transformation (or Laplace tranfrmation)
![image](https://github.com/wangjinlong9788/Solve-Poisson-equation-using-CUDA-FFT/blob/master/step.PNG)

## How to use FFT with CUDA(for example, with 1-D):

Firstly, include headfile for FFT

#include <cuda.h>

#include <cufft.h>

#include "cuda_runtime.h"

#include "device_launch_parameters.h" 

cufftHandle plan; // create cuFFT handle

cufftPlan1d(&plan, N, CUFFT_C2C, BATCH);
// N here is the length of data, CUFFT_C2C is from complex to complex; Batch execution for multiple transforms,or use cufftPlanMany() instead.

cufftExecC2C(plan, data_dev, data_dev, CUFFT_FORWARD); //the first data_dev is the address of input data, and the second  data_dev is address of output data result

// excute cuFFT with forwad FFT, CUFFT_INVERSE is inverse FFT.watch out: InverseFFT needs to diveded by N after execution.
