#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define USE_FISQRT

__device__ float fisqrt(const float number);

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

   __device__ Vec operator +(const Vec& rhs) const
   {
      Vec res;
      res.x = this->x + rhs.x;
      res.y = this->y + rhs.y;

      return res;
   }

   __device__ Vec operator +=(const Vec& rhs)
   {
      this->x += rhs.x;
      this->y += rhs.y;

      return *this;
   }

   __device__ Vec operator -(const Vec& rhs) const
   {
      Vec res;
      res.x = this->x - rhs.x;
      res.y = this->y - rhs.y;

      return res;
   }

   __device__ Vec operator *(const float fac) const
   {
      Vec res;
      res.x = this->x * fac;
      res.y = this->y * fac;

      return res;
   }

   __device__ Vec operator /(const float fac) const
   {
      Vec res;
      res.x = this->x / fac;
      res.y = this->y / fac;

      return res;
   }

   __device__ Vec& operator *=(const float fac)
   {
      this->x *= fac;
      this->y *= fac;

      return *this;
   }

   __device__ Vec& operator /=(const float fac)
   {
      if(fac != 0)
      {
         this->x /= fac;
         this->y /= fac;
      }
      else
      {
         this->x /= 0;
         this->y /= 0;
      }

      return *this;
   }

   __device__ float lengthSquared() const
   {
      return this->x * this->x + this->y * this->y;
   }

   __device__ float getLength() const
   {
      return sqrtf(this->x * this->x + this->y * this->y);
   }

   __device__  Vec normalized() const
   {

#if defined(USE_FISQRT)

      Vec res;
      float length_squared = lengthSquared();

      if(length_squared != 0)
      {
         float inv_sqrt_length = fisqrt(length_squared);
         res.x = this->x * inv_sqrt_length;
         res.y = this->y * inv_sqrt_length;
      }

      return res;

#else

      Vec res;
      float length = getLength();

      if(length)
      {
         res.x = this->x / length;
         res.y = this->y / length;
      }

      return res;

#endif
      
   }

   __device__ void normalize()
   {

#if defined(USE_FISQRT)

      float length_squared = lengthSquared();
      if(length_squared)
      {
         float inv_sqrt_length = fisqrt(length_squared);
         this->x *= inv_sqrt_length;
         this->y *= inv_sqrt_length;
      }

#else

      float length = getLength();

      if(length != 0)
      {
         this->x /= length;
         this->y /= length;
      }

#endif

   }
   __device__ void limit(const float limitLength)
   {
#if defined(USE_FISQRT)

      float length_squared = lengthSquared();
      if(length_squared && length_squared > limitLength * limitLength)
      {
         float inv_sqrt_length = fisqrt(length_squared) * limitLength;
         this->x *= inv_sqrt_length;
         this->y *= inv_sqrt_length;
      }

#else

      float length = getLength();

      if(length != 0 && length > limitLength)
      {
         this->x = this->x / length * limitLength;
         this->y = this->y / length * limitLength;
      }

#endif

   }

   __device__ static Vec direction(const Vec& from, const Vec& to)
   {
      Vec res = to - from;
      return res.normalized();
   }

   __device__ static float distanceSquared(const Vec& vec1, const Vec& vec2)
   {
      return (vec1 - vec2).lengthSquared();
   }

   __device__ void setLength(const float newLength)
   {
      normalize();
      this->x *= newLength;
      this->y *= newLength;
   }

   __device__ Vec perp() const
   {
      return Vec(-y, x);
   }
};

__device__ float uintTo01(const unsigned int i);

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

   __device__ void updatePosition(const float width, const float height)
   {
      const float newX = pos.x + vel.x;
      const float newY = pos.y + vel.y;

      const bool checkX = newX < 0 || newX >= width;
      const bool checkY = newY < 0 || newY >= height;

      vel.x *= -1 * checkX + 1 * !checkX;
      vel.y *= -1 * checkY + 1 * !checkY;

      pos += vel;

      /*if(pos.x < 0)
      {
         vel.x *= -1;
         pos.x = 0;
      }
      else
      {
         if(pos.x >= width)
         {
            vel.x *= -1;
            pos.x = width - 1;
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
            pos.y = height - 1;
         }
      }*/
   }

   __device__ void steer(const Vec& target, const float force)
   {
      Vec temp = target - pos;
      temp.setLength(force);
      vel += temp;
   }

   __device__ void wander(const unsigned int seed, const float strength)
   {
      vel += vel.perp() * (uintTo01(seed) * 2 - 1) * strength;
   }
};