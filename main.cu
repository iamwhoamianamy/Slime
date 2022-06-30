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

const int AGENTS_COUNT = 800000;
const int DIM_X = 1920;
const int DIM_Y = 1016;

const int GRID_SIZE = DIM_X * DIM_Y;
const int IMAGE_SIZE = DIM_X * DIM_Y * 3;

const int FPS = 45;

__device__ const float SPEED = 0.3f;
__device__ const float MAX_TEMP = 1.0f;
__device__ const float MIN_TEMP = 0.0001f;

__device__ const float RAD = 0.01745329251f;
__device__ const float TRACE_LENGTH = 0.95f;
__device__ const float MAX_VELOCITY = 2.5f;
__device__ const float WANDERING_STRENGTH = 0.6f;
__device__ const float STEERING_FORCE = 3.00f;
__device__ const float PERCEPTION_LENGTH = 6;

__device__ const int FOW_WIDTH = 7;


//float mouseX = DIM_X / 2, mouseY = DIM_Y / 2;

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

//struct Lock
//{
//   int* mutex;
//
//   Lock()
//   {
//      int state = 0;
//      handleCudaMalloc((void**)&mutex, sizeof(int));
//      handleError(cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice));
//   }
//
//   ~Lock()
//   {
//      cudaFree(mutex);
//   }
//
//   __device__ void lock()
//   {
//      while(atomicCAS(mutex, 0, 1));
//   }
//
//   __device__ void unlock()
//   {
//      atomicExch(mutex, 0);
//   }
//};

__global__ void initRandomGenerator(curandState* state)
{
   int id = threadIdx.x + blockIdx.x * blockDim.x;
   curand_init(1234, id, 0, &state[id]);
}

__device__ void setPixel(unsigned char* ptr, int offset, const unsigned char MAX_SPAWN_RADIUS, const unsigned char g, const unsigned char b)
{
   offset *= 3;
   ptr[offset + 0] = MAX_SPAWN_RADIUS;
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

#define myMin(a, b) (a < b) ? a : b

__device__ const float MAX_SPAWN_RADIUS = myMin(DIM_X / 4, DIM_Y / 4);
__device__ const int RINGS_COUNT = 100;

__global__ void initAgents(Agent* agents)
{
   int i = threadIdx.x + blockIdx.x * blockDim.x;

   if(i < AGENTS_COUNT)
   {
      const float step = 360.0f / AGENTS_COUNT;
      const float RING_DIFF = MAX_SPAWN_RADIUS / RINGS_COUNT;

      const float ring_radius = RING_DIFF * (int(i / (float(AGENTS_COUNT) / RINGS_COUNT)));
      Vec pos = Vec(DIM_X / 2 + ring_radius * cos(i * RINGS_COUNT * RAD * step), DIM_Y / 2 + ring_radius * sin(i * RINGS_COUNT * RAD * step));
      Vec vel = Vec(DIM_X / 2 - pos.x, DIM_Y / 2 - pos.y);
      vel.limit(MAX_VELOCITY);
      agents[i] = Agent(pos, vel);
   }
}

__device__ void weightOfRegion(Vec& offset, const float* canvas, float& weight)
{
   if(offset.x >= 0 && offset.x < DIM_X &&
      offset.y >= 0 && offset.y < DIM_Y)
      atomicAdd(&weight, canvas[int(offset.x + offset.y * DIM_X)]);
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
      center = agents[agent_index].pos + agents[agent_index].vel.normalized() * PERCEPTION_LENGTH;
      center_int.x = int(center.x);
      center_int.y = int(center.y);

      center_start = center_int - Vec(int(FOW_WIDTH / 2), int(FOW_WIDTH / 2));
      target = Vec();
   }

   __syncthreads();

   //if(center_start.x >= 0 && center_start.x + fowWidth <= DIM &&
   //   center_start.y >= 0 && center_start.y + fowWidth <= DIM)
   {
      Vec offset = center_start + Vec(threadIdx.x, threadIdx.y);

      if(offset.x >= 0 && offset.x < DIM_X &&
         offset.y >= 0 && offset.y < DIM_Y)
      {
         Vec temp = Vec::direction(center_int, offset) * canvas[int(offset.x + offset.y * DIM_X)];

         atomicAdd(&(target.x), temp.x);
         atomicAdd(&(target.y), temp.y);
      }
   }

   __syncthreads();

   if(threadIdx.x == 0 && threadIdx.y == 0)
   {
      if(target.lengthSquared() > 1e-7)
      {
         agents[agent_index].steer(center + target, STEERING_FORCE);
      }
   }
}

__global__ void updateAgents(Agent* agents, curandState* state)
{
   const int offset = threadIdx.x + blockIdx.x * blockDim.x;
   curandState localState = state[offset];

   if(offset < AGENTS_COUNT)
   {
      agents[offset].vel.limit(MAX_VELOCITY);
      agents[offset].wander(curand(&localState), WANDERING_STRENGTH);
      agents[offset].updatePosition(DIM_X, DIM_Y);
   }

   state[offset] = localState;
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

   if(offset < GRID_SIZE)
   {
      int left = (offset - 1) * (x > 0) + offset * (x <= 0);
      int right = (offset + 1) * (x + 1 < DIM_X) + offset * (x + 1 >= DIM_X);

      int bot = (offset - DIM_X) * (y > 0) + offset * (y <= 0);
      int top = (offset + DIM_X) * (y + 1 < DIM_Y) + offset * (y + 1 >= DIM_Y);

      /*int left = x > 0 ? offset - 1 : offset;
      int right = x + 1 < DIM_X ? offset + 1 : offset;

      int bot = y > 0 ? offset - DIM_X : offset;
      int top = y + 1 < DIM_Y ? offset + DIM_X : offset;*/

      //outSrc[offset] = inSrc[offset] + SPEED * (inSrc[top] + inSrc[bot] + inSrc[left] + inSrc[right] - 4 * inSrc[offset]);
      outSrc[offset] = TRACE_LENGTH * (inSrc[top] + inSrc[bot] + inSrc[left] + inSrc[right]) / 4.0f;
   }
}

__global__ void floatToColor(const float* values, unsigned char* colors)
{
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;
   int offset = x + y * blockDim.x * gridDim.x;

   if(offset < GRID_SIZE)
   {
      setPixel(colors, offset, (unsigned char)(values[offset] * 255));
   }
}

handler freeMemory();

//curandGenerator_t gen;

void initMemory()
{
   handleCudaMalloc((void**)&data.devPixels, IMAGE_SIZE * sizeof(float));
   handleCudaMalloc((void**)&data.devInSrc, GRID_SIZE * sizeof(float));
   handleCudaMalloc((void**)&data.devOutSrc, GRID_SIZE * sizeof(float));
   handleCudaMalloc((void**)&data.devAgents, AGENTS_COUNT * sizeof(Agent));
   handleCudaMalloc((void**)&data.devState, AGENTS_COUNT * sizeof(curandState));

   initAgents<<<(AGENTS_COUNT + 3) / 4, 4>>>(data.devAgents);
   handleKernelLaunch();
   handleKernelSynchronize();

   initRandomGenerator<<<(AGENTS_COUNT + 3) / 4, 4 >>>(data.devState);
   handleKernelLaunch();
   handleKernelSynchronize();

   data.output_pixels = new unsigned char[IMAGE_SIZE];

   //curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
   //curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
}

const int REPEATS = 1;

const int CHK_TPB = 32;
const dim3 COPY_HEATERS_KERNEL_BLOCKS((AGENTS_COUNT + CHK_TPB - 1) / CHK_TPB);
const dim3 COPY_HEATERS_KERNEL_THREADS(CHK_TPB);

const int BK_TPB = 32;

const dim3 BLEND_KERNEL_BLOCKS((DIM_X + BK_TPB - 1) / BK_TPB,
                  (DIM_Y + BK_TPB - 1) / BK_TPB);

const dim3 BLEND_KERNEL_THREADS(BK_TPB, BK_TPB);

__global__ void copyHeatersKernel(float* inPtr, const Agent* agents);

void formHeatmap()
{
   for(size_t i = 0; i < REPEATS; i++)
   {
      copyHeatersKernel<<<COPY_HEATERS_KERNEL_BLOCKS, COPY_HEATERS_KERNEL_THREADS >>>(data.devInSrc, data.devAgents);
      handleKernelLaunch();
      handleKernelSynchronize();

      blendKernel<<<BLEND_KERNEL_BLOCKS, BLEND_KERNEL_THREADS >>>(data.devOutSrc, data.devInSrc);
      handleKernelLaunch();
      handleKernelSynchronize();

      std::swap(data.devInSrc, data.devOutSrc);
   }

   floatToColor<<<BLEND_KERNEL_BLOCKS, BLEND_KERNEL_THREADS>>>(data.devInSrc, data.devPixels);
   handleKernelLaunch();
   handleKernelSynchronize();

   handleError(cudaMemcpy(data.output_pixels, data.devPixels, IMAGE_SIZE * sizeof(unsigned char), cudaMemcpyDeviceToHost));
}

__global__ void copyHeatersKernel(float* inPtr, const Agent* agents)
{
   __shared__ Vec agentPos[CHK_TPB];
   const unsigned int offset = threadIdx.x + blockIdx.x * blockDim.x;

   agentPos[threadIdx.x] = agents[threadIdx.x + blockIdx.x * blockDim.x].pos;

   __syncthreads();

   inPtr[(int)agentPos[threadIdx.x].x + (int)agentPos[threadIdx.x].y * DIM_X] = MAX_TEMP;

   __syncthreads();

      /*const unsigned int offset = threadIdx.x + blockIdx.x * blockDim.x;

      const unsigned int x = int(agents[offset].pos.x);
      const unsigned int y = int(agents[offset].pos.y);

      const unsigned int in_offset = x + y * DIM_X;

      if(offset < AGENTS_COUNT)
      {
         inPtr[in_offset] = MAX_TEMP;
      }*/
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
   glViewport(0, 0, DIM_X, DIM_Y);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glOrtho(0, DIM_X, 0, DIM_Y, -1.0, 1.0);
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

const dim3 FOW_THREADS(FOW_WIDTH, FOW_WIDTH);

// Функция вывода на экран 
void display()
{
   glClear(GL_COLOR_BUFFER_BIT);

   glRasterPos2i(0, 0);
   formHeatmap();
   glDrawPixels(DIM_X, DIM_Y, GL_RGB, GL_UNSIGNED_BYTE, data.output_pixels);

   //followPathSample<<<AGENTS_COUNT, fowThreads>>>(data.devAgents, data.devInSrc);
   followPathDensity<<<AGENTS_COUNT, FOW_THREADS>>>(data.devAgents, data.devInSrc);
   handleKernelLaunch();
   handleKernelSynchronize();

   updateAgents << <(AGENTS_COUNT + 4) / 4, 4>> > (data.devAgents, data.devState);
   handleKernelLaunch();
   handleKernelSynchronize();

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
   handleCudaReset();
   std::cout << "Done!";
}

__device__ float uintTo01(const unsigned int i)
{
   return float(i) / UINT_MAX;
}

int main(int argc, char** argv)
{

   glutInit(&argc, argv);
   glutInitDisplayMode(GLUT_RGB);
   glutInitWindowSize(DIM_X, DIM_Y);
   glutCreateWindow("Слизни");
   //std::cin.get();

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

__device__ float fisqrt(const float number)
{
   long i;
   float x2, y;
   const float threehalfs = 1.5F;

   x2 = number * 0.5F;
   y = number;
   i = *(long*)&y;                       // evil floating point bit level hacking
   i = 0x5f3759df - (i >> 1);               // what the fuck? 
   y = *(float*)&i;
   y = y * (threehalfs - (x2 * y * y));   // 1st iteration
   //	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

   return y;
}

//__global__ void followPathSample(Agent* agents, const float* canvas)
//{
//   __shared__ int agent_index;
//
//   __shared__ Vec center;
//   __shared__ Vec left;
//   __shared__ Vec right;
//
//   __shared__ float center_weight;
//   __shared__ float left_weight;
//   __shared__ float right_weight;
//
//   __shared__ Vec center_start;
//   __shared__ Vec left_start;
//   __shared__ Vec right_start;
//
//   if(threadIdx.x == 0 && threadIdx.y == 0)
//   {
//      agent_index = blockIdx.x;
//
//      Vec temp = agents[agent_index].vel.normalized() * perception / 2;
//
//      center = agents[agent_index].pos + temp * 2;
//      left = agents[agent_index].pos + temp + temp.perp();
//      right = agents[agent_index].pos + temp - temp.perp();
//
//      center_weight = 0;
//      left_weight = 0;
//      right_weight = 0;
//
//      center_start = Vec(center.x - fowWidth / 2 - 1, center.y - fowWidth / 2 - 1);
//      left_start = Vec(left.x - fowWidth / 2 - 1, left.y - fowWidth / 2 - 1);
//      right_start = Vec(right.x - fowWidth / 2 - 1, right.y - fowWidth / 2 - 1);
//   }
//   __syncthreads();
//
//   weightOfRegion(center_start + Vec(threadIdx.x, threadIdx.y), canvas, center_weight);
//   weightOfRegion(left_start + Vec(threadIdx.x, threadIdx.y), canvas, left_weight);
//   weightOfRegion(right_start + Vec(threadIdx.x, threadIdx.y), canvas, right_weight);
//
//   __syncthreads();
//
//   if(threadIdx.x == 0 && threadIdx.y == 0)
//   {
//      if(left_weight > center_weight && left_weight > right_weight)
//      {
//         agents[agent_index].steer(left, steeringForce);
//      }
//      else if(right_weight > center_weight && right_weight > left_weight)
//      {
//         agents[agent_index].steer(right, steeringForce);
//      }
//   }
//}
