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

const int AGENTS_COUNT = 10;
const int DIM = 512;
const int GRID_SIZE = DIM * DIM;
const int IMAGE_SIZE = DIM * DIM * 3;
const int FPS = 60;
__device__ const float SPEED = 0.25f;
__device__ const float MAX_TEMP = 1.0f;
__device__ float rad = 0.01745329251f;

const float MIN_TEMP = 0.0001f;

const dim3 blocks(DIM / 16, DIM / 16);
const dim3 threads(16, 16);

float mouseX, mouseY;

struct DataBlock
{
   unsigned char* output_pixels;
   unsigned char* dev_pixels;
   float* dev_inSrc;
   float* dev_outSrc;
   Agent* dev_agents;
};

DataBlock data;

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
      Vec vel = Vec(i % 2 ? 1 : -1, i % 4 ? 1 : -1);
      vel *= 0.5f + i * (1.0f / AGENTS_COUNT);
      agents[i] = Agent(pos, vel);
      //agents[i] = Agent(Vec(DIM / (AGENTS_COUNT - 1) * i, 200), Vec(i % 2 ? 1 : -1, i % 4 ? 1 : -1));
   }
      //agents[i] = Agent(Vec(DIM / (AGENTS_COUNT - 1) * i, 200), Vec(i % 2 ? 1 : -1, i % 4 ? 1 : -1));
}

__global__ void updateAgents(Agent* agents)
{
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   if(i < AGENTS_COUNT)
      agents[i].updatePosition(DIM, DIM);
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
   outSrc[offset] = 0.999f * (inSrc[top] + inSrc[bot] + inSrc[left] + inSrc[right]) / 4.0f;
}

__global__ void floatToColor(const float* values, unsigned char* colors)
{
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;
   int offset = x + y * blockDim.x * gridDim.x;

   setPixel(colors, offset, (unsigned char)(values[offset] * 255));
}

handler freeMemory();

void initMemory()
{
   tryCudaMalloc((void**)&data.dev_pixels, IMAGE_SIZE * sizeof(float));
   tryCudaMalloc((void**)&data.dev_inSrc, GRID_SIZE * sizeof(float));
   tryCudaMalloc((void**)&data.dev_outSrc, GRID_SIZE * sizeof(float));
   tryCudaMalloc((void**)&data.dev_agents, AGENTS_COUNT * sizeof(Agent));

   initAgents<<<(AGENTS_COUNT + 4) / 4, 4>>>(data.dev_agents);
   tryKernelLaunch();
   tryKernelSynchronize();

   data.output_pixels = new unsigned char[IMAGE_SIZE];   
}

const int REPEATS = 10;

void formHeatmap()
{
   for(size_t i = 0; i < REPEATS; i++)
   {
      copyHeatersKernel << <AGENTS_COUNT, 1 >> > (data.dev_inSrc, data.dev_agents);
      tryKernelLaunch();
      tryKernelSynchronize();

      blendKernel << <blocks, threads >> > (data.dev_outSrc, data.dev_inSrc);
      tryKernelLaunch();
      tryKernelSynchronize();

      std::swap(data.dev_inSrc, data.dev_outSrc);
   }

   floatToColor << <blocks, threads >> > (data.dev_inSrc, data.dev_pixels);
   tryKernelLaunch();
   tryKernelSynchronize();

   handleError(cudaMemcpy(data.output_pixels, data.dev_pixels, IMAGE_SIZE * sizeof(unsigned char), cudaMemcpyDeviceToHost));
}

handler freeMemory()
{
   cudaFree(data.dev_pixels);
   cudaFree(data.dev_inSrc);
   cudaFree(data.dev_outSrc);
   cudaFree(data.dev_agents);

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
void Mouse(int button, int state, int x, int y)
{
   mouseX = x;
   mouseY = y;

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

// Функция вывода на экран 
void display()
{
   glClear(GL_COLOR_BUFFER_BIT);

   /*glPointSize(10);
   glBegin(GL_POINTS);

   for(size_t i = 0; i < 10; i++)
   {
      glColor3ub(255, 0, 0);
      glVertex2f((data.dev_agents[i].pos.x), (data.dev_agents[i].pos.y));
   }

   glEnd();*/

   glRasterPos2i(0, 0);
   formHeatmap();
   glDrawPixels(DIM, DIM, GL_RGB, GL_UNSIGNED_BYTE, data.output_pixels);

   //for(size_t i = 0; i < 10; i++)
   //{
   //   data.agents[i].updatePosition(DIM, DIM);
   //}

   updateAgents << <(AGENTS_COUNT + 4) / 4, 4>> > (data.dev_agents);
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
   //glutMouseFunc(Mouse);
   atexit(exitingFunction);
   glutTimerFunc(0, onTimer, 0);

   initMemory();
   glutMainLoop();

   return 0;
}