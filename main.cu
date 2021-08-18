#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "GL/freeglut.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudahelper.cuh"
#include "math_constants.h"
#include "Agent.h"

#include "curand_kernel.h"
//
//#define __cplusplus
//#define __CUDACC__

//#include "device_atomic_functions.hpp"
//#include "device_functions.h"

const int AGENTS_COUNT = 100000;
const int DIM = 2048;
const int GRID_SIZE = DIM * DIM;
const int IMAGE_SIZE = DIM * DIM * 3;
const int FPS = 60;
__device__ const float SPEED = 0.25f;
__device__ const float MAX_TEMP = 1.0f;
__device__ const float MIN_TEMP = 0.0001f;
__device__ float rad = 0.01745329251f;
__device__ float traceLength = 0.98f;
__device__ float maxVelocity = 1.0f;

const dim3 blocks(DIM / 16, DIM / 16);
const dim3 threads(16, 16);

float mouseX = DIM / 2, mouseY = DIM / 2;

struct DataBlock
{
   unsigned char* output_pixels;
   unsigned char* devPixels;
   float* devInSrc;
   float* devOutSrc;
   Agent* devAgents;
   curandState* devState;
};

DataBlock data;

__global__ void initRandomGenerator(curandState* state)
{
   int id = threadIdx.x + blockIdx.x * blockDim.x;
   /* Each thread gets same seed, a different sequence
      number, no offset */
   curand_init(1234, id, 0, &state[id]);
}

__device__ void setPixel(unsigned char* ptr, int offset, const unsigned char r, const unsigned char g, const unsigned char b)
{
   offset *= 3;
   ptr[offset + 0] = r;
   ptr[offset + 1] = g;
   ptr[offset + 2] = b;
}

__device__ void setPixel(unsigned char* ptr, int offset, const unsigned char value)
{
   offset *= 3;
   ptr[offset + 0] = value;
   ptr[offset + 1] = value;
   ptr[offset + 2] = value;
}

__global__ void initAgents(Agent* agents)
{
   int i = threadIdx.x + blockIdx.x * blockDim.x;

   const float r = DIM / 4;
   const float step = 360.0f / AGENTS_COUNT;

   if(i < AGENTS_COUNT)
   {
      Vec pos = Vec(DIM / 2 + r * cos(i * rad * step), DIM / 2 + r * sin(i * rad * step));
      //Vec vel = Vec(i % 2 ? 1 : -1, i % 4 ? 1 : -1);
      //vel *= 0.5f + i * (1.0f / AGENTS_COUNT);
      Vec vel = Vec(DIM / 2 - pos.x, DIM / 2 - pos.y);
      vel.limit(maxVelocity);
      agents[i] = Agent(pos, vel);
      //agents[i] = Agent(Vec(DIM / (AGENTS_COUNT - 1) * i, 200), Vec(i % 2 ? 1 : -1, i % 4 ? 1 : -1));
   }
      //agents[i] = Agent(Vec(DIM / (AGENTS_COUNT - 1) * i, 200), Vec(i % 2 ? 1 : -1, i % 4 ? 1 : -1));
}

__device__ const float steeringForce = 20.75f;
__device__ const float perception = 9;
__device__ const int fowWidth = 11;

__device__ void weightOfRegion(Vec& offset, const float* canvas, float& weight)
{
   if(offset.x >= 0 && offset.x < DIM &&
      offset.y >= 0 && offset.y < DIM)
      atomicAdd(&weight, canvas[int(offset.x + offset.y * DIM)]);
}

__global__ void followPathSample(Agent* agents, const float* canvas)
{
   __shared__ int agent_index;

   __shared__ Vec center;
   __shared__ Vec left;
   __shared__ Vec right;

   __shared__ float center_weight;
   __shared__ float left_weight;
   __shared__ float right_weight;

   __shared__ Vec center_start;
   __shared__ Vec left_start;
   __shared__ Vec right_start;

   if(threadIdx.x == 0 && threadIdx.y == 0)
   {
      agent_index = blockIdx.x;

      Vec temp = agents[agent_index].vel.normalized() * perception / 2;

      center = agents[agent_index].pos + temp * 2;
      left = agents[agent_index].pos + temp + temp.perp();
      right = agents[agent_index].pos + temp - temp.perp();

      center_weight = 0;
      left_weight = 0;
      right_weight = 0;

      center_start = Vec(center.x - fowWidth / 2 - 1, center.y - fowWidth / 2 - 1);
      left_start = Vec(left.x - fowWidth / 2 - 1, left.y - fowWidth / 2 - 1);
      right_start = Vec(right.x - fowWidth / 2 - 1, right.y - fowWidth / 2 - 1);
   }
   __syncthreads();

   weightOfRegion(center_start + Vec(threadIdx.x, threadIdx.y), canvas, center_weight);
   weightOfRegion(left_start + Vec(threadIdx.x, threadIdx.y), canvas, left_weight);
   weightOfRegion(right_start + Vec(threadIdx.x, threadIdx.y), canvas, right_weight);

   __syncthreads();

   if(threadIdx.x == 0 && threadIdx.y == 0)
   {
      if(left_weight > center_weight && left_weight > right_weight)
      {
         agents[agent_index].steer(left, steeringForce);
      }
      else if(right_weight > center_weight && right_weight > left_weight)
      {
         agents[agent_index].steer(right, steeringForce);
      }
   }
}

__global__ void followPathDensity(Agent* agents, const float* canvas)
{
   __shared__ int agent_index;
   __shared__ Vec center;
   __shared__ Vec center_int;
   __shared__ Vec center_start;
   __shared__ Vec target;

   if(threadIdx.x == 0 && threadIdx.y == 0)
   {
      agent_index = blockIdx.x;
      center = agents[agent_index].pos + agents[agent_index].vel.normalized() * perception;
      center_int.x = int(center.x);
      center_int.y = int(center.y);

      center_start = center_int - Vec(int(fowWidth / 2), int(fowWidth / 2));
      target = Vec();
   }

   __syncthreads();

   if(center_start.x >= 0 && center_start.x + fowWidth <= DIM &&
      center_start.y >= 0 && center_start.y + fowWidth <= DIM)
   {
      Vec offset = center_start + Vec(threadIdx.x, threadIdx.y);

      //if(offset.x >= 0 && offset.x < DIM &&
      //   offset.y >= 0 && offset.y < DIM)
      {
         Vec temp = Vec::direction(center_int, offset) * canvas[int(offset.x + offset.y * DIM)];

         atomicAdd(&(target.x), temp.x);
         atomicAdd(&(target.y), temp.y);
      }
   }

   __syncthreads();

   if(threadIdx.x == 0 && threadIdx.y == 0)
   {
      //target /= fowWidth * fowWidth;

      //if(Vec::distanceSquared(center, target))
      if(target.lengthSquared() > 1e-5)
      {
         agents[agent_index].steer(center + target.normalized(), steeringForce);
         agents[agent_index].vel.limit(maxVelocity);
      }
   }
}

__device__ const float wanderingStrength = 0.2f;

__global__ void updateAgents(Agent* agents, curandState* state)
{
   int id = threadIdx.x + blockIdx.x * blockDim.x;
   curandState localState = state[id];

   if(id < AGENTS_COUNT)
   {
      agents[id].wander(curand(&localState), wanderingStrength);
      agents[id].updatePosition(DIM, DIM);
      agents[id].vel.limit(maxVelocity);
   }

   state[id] = localState;
}

__global__ void copyHeatersKernel(float* inPtr, const Agent* agents)
{
   int offset = threadIdx.x + blockIdx.x * blockDim.x;

   int x = int(agents[offset].pos.x);
   int y = int(agents[offset].pos.y);

   int in_offset = x + y * DIM;

   inPtr[in_offset] = MAX_TEMP;
}

__global__ void clearArray(float* arr)
{
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;
   int offset = x + y * blockDim.x * gridDim.x;

   arr[offset] = 0;
}

__global__ void blendKernel(float* outSrc, const float* inSrc)
{
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;
   int offset = x + y * blockDim.x * gridDim.x;

   int left = x > 0 ? offset - 1 : offset;
   int right = x + 1 < DIM ? offset + 1 : offset;

   int bot = y > 0 ? offset - DIM : offset;
   int top = y + 1 < DIM ? offset + DIM : offset;

   //outSrc[offset] = inSrc[offset] + SPEED * (inSrc[top] + inSrc[bot] + inSrc[left] + inSrc[right] - 4 * inSrc[offset]);
   outSrc[offset] = traceLength * (inSrc[top] + inSrc[bot] + inSrc[left] + inSrc[right]) / 4.0f;
}

__global__ void floatToColor(const float* values, unsigned char* colors)
{
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;
   int offset = x + y * blockDim.x * gridDim.x;

   setPixel(colors, offset, (unsigned char)(values[offset] * 255));
}

handler freeMemory();

//curandGenerator_t gen;

void initMemory()
{
   tryCudaMalloc((void**)&data.devPixels, IMAGE_SIZE * sizeof(float));
   tryCudaMalloc((void**)&data.devInSrc, GRID_SIZE * sizeof(float));
   tryCudaMalloc((void**)&data.devOutSrc, GRID_SIZE * sizeof(float));
   tryCudaMalloc((void**)&data.devAgents, AGENTS_COUNT * sizeof(Agent));
   tryCudaMalloc((void**)&data.devState, AGENTS_COUNT * sizeof(curandState));

   initAgents<<<(AGENTS_COUNT + 4) / 4, 4>>>(data.devAgents);
   tryKernelLaunch();
   tryKernelSynchronize();

   initRandomGenerator<<<(AGENTS_COUNT + 4) / 4, 4 >>>(data.devState);
   tryKernelLaunch();
   tryKernelSynchronize();

   data.output_pixels = new unsigned char[IMAGE_SIZE];

   //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
   //curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
}

const int REPEATS = 1;

void formHeatmap()
{
   for(size_t i = 0; i < REPEATS; i++)
   {
      copyHeatersKernel << <AGENTS_COUNT, 1 >> > (data.devInSrc, data.devAgents);
      tryKernelLaunch();
      tryKernelSynchronize();

      blendKernel << <blocks, threads >> > (data.devOutSrc, data.devInSrc);
      tryKernelLaunch();
      tryKernelSynchronize();

      std::swap(data.devInSrc, data.devOutSrc);
   }

   floatToColor << <blocks, threads >> > (data.devInSrc, data.devPixels);
   tryKernelLaunch();
   tryKernelSynchronize();

   handleError(cudaMemcpy(data.output_pixels, data.devPixels, IMAGE_SIZE * sizeof(unsigned char), cudaMemcpyDeviceToHost));
}

handler freeMemory()
{
   cudaFree(data.devPixels);
   cudaFree(data.devInSrc);
   cudaFree(data.devOutSrc);
   cudaFree(data.devAgents);
   cudaFree(data.devState);

   delete[] data.output_pixels;
}

// Функция изменения размеров окна
void reshape(GLint w, GLint h)
{
   glViewport(0, 0, DIM, DIM);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(0, DIM, 0, DIM, -1.0, 1.0);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

// Функция обработки сообщений от клавиатуры 1
void keyboardLetters(unsigned char key, int x, int y)
{
   //switch(key)
   //{
   //   case 'r':
   //   {
   //      for(size_t i = 0; i < GRID_SIZE; i++)
   //      {
   //         data.constSrc[i] = 0;
   //      }

   //      handleError(cudaMemcpy(data.dev_constSrc, data.constSrc, GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
   //      break;
   //   }
   //   case 'c':
   //   {
   //      clearArray << <blocks, threads >> > (data.dev_inSrc);
   //      tryKernelLaunch();
   //      tryKernelSynchronize();

   //      break;
   //   }
   //}

   ////glutPostRedisplay();
}

// Функция обработки сообщения от мыши
void mouse(int button, int state, int x, int y)
{


   //// Клавиша была нажата, но не отпущена
   //if(state != GLUT_DOWN) return;

   //// Новая точка по левому клику
   //if(button == GLUT_LEFT_BUTTON)
   //{
   //   data.constSrc[x + DIM * (DIM - y)] = MAX_TEMP;

   //   //cudaFree(data.dev_constSrc);
   //   //tryCudaMalloc((void**)&data.dev_constSrc, GRID_SIZE * sizeof(float));
   //   handleError(cudaMemcpy(data.dev_constSrc, data.constSrc, GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
   //}

   //// Новая точка по левому клику
   //if(button == GLUT_RIGHT_BUTTON)
   //{
   //   data.constSrc[x + DIM * (DIM - y)] = MIN_TEMP;

   //   //cudaFree(data.dev_constSrc);
   //   //tryCudaMalloc((void**)&data.dev_constSrc, GRID_SIZE * sizeof(float));
   //   handleError(cudaMemcpy(data.dev_constSrc, data.constSrc, GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice));
   //}
}

void mousePassive(int x, int y)
{

}

const dim3 fowThreads(fowWidth, fowWidth);

// Функция вывода на экран 
void display()
{
   glClear(GL_COLOR_BUFFER_BIT);

   glRasterPos2i(0, 0);
   formHeatmap();
   glDrawPixels(DIM, DIM, GL_RGB, GL_UNSIGNED_BYTE, data.output_pixels);

   //followPathSample<<<AGENTS_COUNT, fowThreads>>>(data.devAgents, data.devInSrc);
   followPathDensity<<<AGENTS_COUNT, fowThreads>>>(data.devAgents, data.devInSrc);
   tryKernelLaunch();
   tryKernelSynchronize();

   updateAgents << <(AGENTS_COUNT + 4) / 4, 4>> > (data.devAgents, data.devState);
   tryKernelLaunch();
   tryKernelSynchronize();

   glFinish();
}

void onTimer(int millisec)
{
   glutPostRedisplay();
   glutTimerFunc(1000 / FPS, onTimer, 0);
}

void exitingFunction()
{
   freeMemory();
   //tryCudaLastError();
   tryCudaReset();
   std::cout << "Done!";
}

__device__ float uintTo01(const unsigned int i)
{
   float res = float(i) / UINT_MAX;
   return res;
}

int main(int argc, char** argv)
{
   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGB);
   glutInitWindowSize(DIM, DIM);
   glutCreateWindow("Нагрев");

   glShadeModel(GL_FLAT);
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

   glutDisplayFunc(display);
   glutReshapeFunc(reshape);
   //glutKeyboardFunc(keyboardLetters);
   glutMouseFunc(mouse);
   glutPassiveMotionFunc(mousePassive);
   atexit(exitingFunction);
   glutTimerFunc(0, onTimer, 0);

   initMemory();
   glutMainLoop();

   return 0;
}