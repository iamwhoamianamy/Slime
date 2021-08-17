#pragma once
#include "cuda_runtime.h"
#include <cstdlib>
#include <stdio.h>
#include <windows.h>

typedef void handler;
#define FOREGROUND_WHITE 7

void printg(const char *str)
{
   SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN);
   printf(str);
   SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_WHITE);
}

void printr(const char* str)
{
   SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED);
   printf(str);
   SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_WHITE);
}

handler doNothing()
{

}

cudaError_t handleError(cudaError_t functReturn, handler (*errorHandler)() = doNothing)
{
   switch(functReturn)
   {
      case cudaError::cudaSuccess:
      {
         return functReturn;
      }
      default:
      {
         printr("ERROR!\n");
         errorHandler();
         std::exit(0);
      }
   }
}

void tryCudaMalloc(void** devPtr, const size_t size, handler(*errorHandler)() = doNothing)
{
   cudaError_t cudaStatus = cudaMalloc(devPtr, size);

   if(cudaStatus != cudaError::cudaSuccess)
   {
      printr("Error on allocating memory!\n");
      errorHandler();
      std::exit(0);
   }
}

void tryCudaLastError()
{
   cudaError_t cudaStatus = cudaGetLastError();

   if(cudaStatus != cudaError::cudaSuccess)
   {
      printr("An error occured!\n");

      std::exit(0);
   }
}

void tryCudaReset()
{
   cudaError_t cudaStatus = cudaDeviceReset();

   if(cudaStatus == cudaError::cudaSuccess)
   {
      printg("Cuda device reset success!\n");
   }
   else
   {
      printr("Cuda device reset failed!\n");

      std::exit(0);
   }
}

void tryKernelLaunch(handler(*errorHandler)() = doNothing)
{
   cudaError_t cudaStatus = cudaGetLastError();

   if(cudaStatus != cudaError::cudaSuccess)
   {
      printr("Kernel launch failed!\n");
      errorHandler();
      std::exit(0);
   }
}

void tryKernelSynchronize(handler(*errorHandler)() = doNothing)
{
   cudaError_t cudaStatus = cudaDeviceSynchronize();

   if(cudaStatus != cudaError::cudaSuccess)
   {
      printr("Kernel synchrinisation failed!\n");
      errorHandler();
      std::exit(0);
   }
}


