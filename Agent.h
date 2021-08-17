#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ struct Vec
{
   float x;
   float y;

   __device__ Vec() : x(0), y(0) {}
   __device__ Vec(float x, float y) : x(x), y(y) {}
   //__device__ Vec(float mag) : x(float(rand()) / RAND_MAX * mag), y(float(rand()) / RAND_MAX * mag) {}

   //__device__ static Vec& random(float xMin, float xMax, float yMin, float yMax)
   //{
   //   Vec res;
   //   res.x = float(rand()) / RAND_MAX * (xMax - xMin) + xMin;
   //   res.y = float(rand()) / RAND_MAX * (yMax - yMin) + yMin;
   //   return res;
   //}

   __device__ Vec& operator +(const Vec& rhs) const
   {
      Vec res;
      res.x = this->x + rhs.x;
      res.y = this->y + rhs.y;


      return res;
   }

   __device__ Vec& operator +=(const Vec& rhs)
   {
      this->x += rhs.x;
      this->y += rhs.y;

      return *this;
   }

   __device__ Vec& operator -(const Vec& rhs) const
   {
      Vec res;
      res.x = this->x - rhs.x;
      res.y = this->y - rhs.y;

      return res;
   }

   __device__ Vec& operator *(const float fac) const
   {
      Vec res;
      res.x = this->x * fac;
      res.y = this->y * fac;

      return res;
   }

   __device__ Vec& operator *=(const float fac)
   {
      this->x *= fac;
      this->y *= fac;

      return *this;
   }

   __device__ float lengthSquared() const
   {
      return this->x * this->x + this->y * this->y;
   }

   __device__  Vec& normalized() const
   {
      Vec res;
      float length = sqrtf(lengthSquared());
      res.x = this->x / length;
      res.y = this->y / length;

      return res;
   }

   __device__ void normalize()
   {
      float length = sqrtf(lengthSquared());
      this->x /= length;
      this->y /= length;
   }

   __device__ void setLength(const float newLength)
   {
      normalize();
      this->x *= newLength;
      this->y *= newLength;
   }
};

__device__ class Agent
{
public:
   Vec pos;
   Vec vel;

   __device__ Agent() : pos(), vel() {}
   __device__ Agent(const Vec& pos) : pos(pos), vel() {}
   __device__ Agent(const Vec& pos, const Vec& vel) : pos(pos), vel(vel) {}

   //__device__ static Agent& random(const float width, const float height, const float speed)
   //{
   //   Agent res;
   //   res.pos = Vec::random(0, width, 0, height);
   //   res.vel = Vec(speed);
   //   return res;
   //}

   __device__ void updatePosition(float width, float height)
   {
      pos += vel;
      
      if(pos.x < 0)
      {
         vel.x *= -1;
         pos.x = 0;
      }
      else
      {
         if(pos.x >= width)
         {
            vel.x *= -1;
            pos.x = width;
         }
      }

      if(pos.y < 0)
      {
         vel.y *= -1;
         pos.y = 0;
      }
      else
      {
         if(pos.y >= height)
         {
            vel.y *= -1;
            pos.y = height;
         }
      }
   }

   __device__ void steer(const Vec& vec, const float force)
   {
      Vec temp = vec - pos;
      temp.setLength(force);
      vel += temp;
   }
};