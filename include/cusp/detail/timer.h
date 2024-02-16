// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cuda.h>

namespace cusp
{
namespace detail
{

class timer
{
  public:
    size_t calls;
    bool paused;
    double milliseconds;
    cudaEvent_t _start;
    cudaEvent_t _end;

    timer() : _start(NULL), _end(NULL)
    { 
      reset();
    }

    ~timer()
    {
      stop();
    }

    void operator+=(const timer &b) 
    {
      milliseconds += b.milliseconds;
      calls += b.calls;
    }

    bool is_empty(void)  const { return milliseconds == 0.0; }
    bool is_paused(void) const { return paused; }

    void unpause(void) 
    { 
      cudaEventRecord(_start,0);
      paused = false; 
    }

    void pause(void) 
    { 
      stop();
      cudaEventCreate(&_start); 
      cudaEventCreate(&_end);
      paused = true; 
    }            

    void start(void) 
    { 
      ++calls; 
      cudaEventCreate(&_start); 
      cudaEventCreate(&_end);
      cudaEventRecord(_start,0);
    }

    void stop(void) 
    { 
      if(_start != NULL && _end != NULL )
      {
        milliseconds += milliseconds_elapsed(); 

        cudaEventDestroy(_start);
        cudaEventDestroy(_end);

        _start = NULL;
        _end   = NULL;
      }
    }

    void soft_stop(void) 
    { 
      if ( !paused ) 
        milliseconds = milliseconds_elapsed(); 
    }

    void reset(void) 
    { 
      if(_start != NULL ) cudaEventDestroy(_start);
      if(_end   != NULL ) cudaEventDestroy(_end);

      cudaEventCreate(&_start); 
      cudaEventCreate(&_end);

      calls = 0;
      paused = false;
      milliseconds = 0.0;
    }

    void soft_reset(void) 
    { 
      calls = 0; 
      milliseconds = 0.0; 

      cudaEventDestroy(_start);
      cudaEventDestroy(_end);
      cudaEventCreate(&_start); 
      cudaEventCreate(&_end);
    }

    float milliseconds_elapsed()
    { 
      float elapsed_time;
      cudaEventRecord(_end, 0);
      cudaEventSynchronize(_end);
      cudaEventElapsedTime(&elapsed_time, _start, _end);
      return elapsed_time;
    }

    float seconds_elapsed()
    { 
      return milliseconds_elapsed() / 1000.0;
    }

};

} // end namespace detail
} // end namespace cusp

