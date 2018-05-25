##Solve Poisson equation using CUDA FFT

When using Visual studio, click right button on your project name. Then click on properties. 

Then configuration properties, linker, input. In additional dependencies you must write cufft.lib.

 ![image](https://github.com/wangjinlong9788/Solve-Poisson-equation-using-CUDA-FFT/blob/master/Possion.PNG)


![image](https://github.com/wangjinlong9788/Solve-Poisson-equation-using-CUDA-FFT/blob/master/step.PNG)

How to use FFT with CUDA(for example, with 1-D):

cufftHandle plan; // create cuFFT handle
cufftPlan1d(&plan, N, CUFFT_C2C, BATCH);
cufftExecC2C(plan, data_dev, data_dev, CUFFT_FORWARD); // excute cuFFT with forwad FFT, 
