// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef _WIN32
// For now, on VS10, this is required to avoid a compilation error.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
// #include <time.h>
#endif

#ifdef NVTX_RANGES
#include "nvToolsExt.h"
#endif

#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cassert>

#include <algorithm>
#include <chrono>

namespace amgx
{
class nvtxRange
{
    static int color_counter;

#ifdef NVTX_RANGES
    nvtxRangeId_t id;
#endif

public:
    nvtxRange(const char*, int color = -1);
    ~nvtxRange();
};

/**********************************************
 *  class for holding profiling data if desired
 *********************************************/
typedef std::map<const char *, double> Times;
typedef std::map<const char *, std::chrono::high_resolution_clock::time_point> Event;

class levelProfile
{
    private:
#ifdef PROFILE
        Times times;
        Event Tic;
#endif

    public:
        levelProfile() { }
        ~levelProfile() {}

        inline void tic(const char *event)
        {
#ifdef PROFILE
            cudaDeviceSynchronize();
            Tic[event] = high_resolution_clock::now();
#endif
        }

        inline void toc(const char *event)
        {
#ifdef PROFILE
            cudaDeviceSynchronize();
            std::chrono::duration<double, std::nano> ns = t2 - t1;
            times[event] += ns.count();
#endif
        }

#ifdef PROFILE
        std::vector<const char *>
#else
        void
#endif
        inline getHeaders()
        {
#ifdef PROFILE
            std::vector<const char *> headerVec;

            for (auto it = times.begin(); it != times.end(); ++it)
            {
                headerVec.push_back(it->first);
            }

            return headerVec;
#endif
        }

#ifdef PROFILE
        std::vector<double>
#else
        void
#endif
        inline getTimes()
        {
#ifdef PROFILE
            std::vector<double> res;

            for (auto it = times.begin(); it != times.end(); ++it)
            {
                res.push_back(it->second);
            }

            return times;
#endif
        }

        /********************************************
         * Reset all events
         *******************************************/
        inline void resetTimer()
        {
#ifdef PROFILE

            for (auto it = times.begin(); it != times.end(); ++it)
            {
                it->second = 0.0;
            }

#endif
        }
};


#ifdef AMGX_USE_CPU_PROFILER

class Profiler_entry
{
        static const int WIDTH = 64;
    private:
        char m_name[WIDTH];
        unsigned long long m_time;
        high_resolution_clock::time_point tic;
        std::vector<Profiler_entry> m_children;

    public:
        inline Profiler_entry() : m_time(0)
        {
            m_name[0] = '\0';
            m_children.reserve(32);
        }

        inline Profiler_entry(const char *name) : m_time(0)
        {
            if ( name == NULL )
            {
                m_name[0] = '\0';
            }
            else
            {
                strncpy(m_name, name, WIDTH - 1);
            }

            m_name[WIDTH - 1] = '\0';
        }

        inline Profiler_entry(const Profiler_entry &other) :
            m_time(other.m_time),
            m_children(other.m_children)
        {
            strncpy(m_name, other.m_name, WIDTH);
        }

        inline Profiler_entry &operator = ( const Profiler_entry &other )
        {
            strncpy(m_name, other.m_name, WIDTH);
            m_time = other.m_time;
            m_children = other.m_children;
            return *this;
        }

        inline Profiler_entry *add_child( const char *name )
        {
            m_children.push_back( Profiler_entry() );
            Profiler_entry &back = m_children.back();
            back.set_name(name);
            return &m_children.back();
        }

        inline double get_time_in_ms() const
        {
            return 1.0e-6 * m_time;
        }

        inline unsigned long long get_time_in_ns() const
        {
            return m_time;
        }

        inline void set_time(unsigned long long time)
        {
            m_time = time;
        }

        std::ostream &print(std::ostream &out, int depth, int max_depth, double total_time, double parent_time) const;

        inline void set_name( const char *name )
        {
            strncpy( m_name, name, WIDTH - 1 );
            m_name[WIDTH - 1] = '\0';
        }

        inline void start()
        {
            tic = high_resolution_clock::now();
        }

        inline void stop()
        {
            m_time = std::max( 0ull, (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1)).count());
        }
};

class Profiler_tree
{
        std::vector<Profiler_entry *> m_stack;
        Profiler_entry m_root;
        int m_max_depth;
        high_resolution_clock::time_point m_first_start, m_last_stop;
#ifdef AMGX_WITH_MPI
        int m_rank;
#endif
        std::map<std::string, int> m_markers;

    public:

        static Profiler_tree &get_instance();

        Profiler_tree();
        ~Profiler_tree();

        void push(const char *name);
        void pop();

        void mark(const char *name, const char *text);
};

class Profiler_raii
{
        std::string m_name;
    public:
        Profiler_raii( const char *name, const char *filename, int lineno );
        ~Profiler_raii();
};

#define AMGX_CPU_PROFILER(name) Profiler_raii __profiler_object(name, __FILE__, __LINE__)
#define AMGX_CPU_MARKER(name, text) Profiler_tree::get_instance().mark(name, text)
#define AMGX_CPU_COND_MARKER(cond, name, text) if(cond) Profiler_tree::get_instance().mark(name, text)

#else
#define AMGX_CPU_PROFILER(name)
#define AMGX_CPU_MARKER(name, text)
#define AMGX_CPU_COND_MARKER(cond, name, text)
#endif


// general purpose timer

class Timer
{
    protected:
        double m_accumulated_time;
        int    m_counter;
        bool   m_accumulate_average;
    public:
        Timer(bool accumulate_average): m_accumulated_time(0.0), m_counter(0), m_accumulate_average(accumulate_average) {};
        virtual ~Timer() {};

        virtual void start() = 0;
        virtual double stop() = 0;
        virtual double elapsed() = 0;

        virtual bool is_running() = 0;

        double get_accumulated_time()
        {
            return m_accumulate_average ? (m_accumulated_time / (m_counter > 0 ? m_counter : 1)) : m_accumulated_time;
        }
};

class TimerCPU : public Timer
{
    std::chrono::high_resolution_clock::time_point  m_last_tick;
        bool    m_is_running;
    public:
        TimerCPU(bool accumulate_average): Timer(accumulate_average), m_is_running(false) {};
        ~TimerCPU()
        {
            this->stop();
        };

        void start()
        {
            if (!m_is_running)
            {
                m_last_tick = std::chrono::high_resolution_clock::now();
                m_is_running = true;
            }
        }

        double stop()
        {
            double res = 0.0;

            if (m_is_running)
            {
                std::chrono::duration<double> dsec = std::chrono::high_resolution_clock::now() - m_last_tick;
                res = (std::chrono::duration_cast<std::chrono::nanoseconds>(dsec)).count();
                m_is_running = false;
                m_counter ++;
            }

            m_accumulated_time += res;
            return res;
        }

        double elapsed()
        {
            double res = 0.0;

            if (m_is_running)
            {
                std::chrono::duration<double> dsec = std::chrono::high_resolution_clock::now() - m_last_tick;
                res = (std::chrono::duration_cast<std::chrono::nanoseconds>(dsec)).count();
            }

            return res;
        }

        bool is_running()
        {
            return m_is_running;
        }

};


class TimerGPU_Events : public Timer
{
        cudaEvent_t m_start, m_stop, m_inter;
        bool        m_is_running;
    public:
        TimerGPU_Events(bool accumulate_average): Timer(accumulate_average), m_is_running(false)
        {
            cudaEventCreate(&m_start);
            cudaEventCreate(&m_stop);
        }

        ~TimerGPU_Events()
        {
            this->stop();
            cudaEventDestroy(m_start);
            cudaEventDestroy(m_stop);
        }

        void start()
        {
            if (!m_is_running)
            {
                cudaEventRecord(m_start);
                m_is_running = true;
            }
        }

        double stop()
        {
            float res = 0.0f;

            if (m_is_running)
            {
                cudaEventRecord(m_stop);
                cudaEventSynchronize(m_stop);
                cudaEventElapsedTime( &res, m_start, m_stop);
                res *= 1e-3f;
                m_accumulated_time += static_cast<double>(res);
                m_is_running = false;
                m_counter ++;
            }

            return static_cast<double>(res);
        }

        double elapsed()
        {
            float res = 0.0f;

            if (m_is_running)
            {
                cudaEventRecord(m_stop);
                cudaEventSynchronize(m_stop);
                cudaEventElapsedTime( &res, m_start, m_stop);
                res *= 1e-3f;
            }

            return static_cast<double>(res);
        }

        bool is_running()
        {
            return m_is_running;
        }
};

typedef enum
{
    CPU_TIMER = 1,
    GPU_TIMER = 2,
    CREATE_AND_START = 4,
    ACCUMULATE_AVERAGE = 8,
} TIMER_FLAGS;

class TimerMap
{
    private:
        std::map<std::string, Timer *> m_timers;
    public:
        TimerMap() {};
        ~TimerMap()
        {
            for (std::map<std::string, Timer *>::iterator iter = m_timers.begin(); iter != m_timers.end(); ++iter)
            {
                delete iter->second;
            }
        };

        int createTimer(const char *label, unsigned int flags)
        {
            Timer *timer;

            if ( ((flags & CPU_TIMER) && (flags & GPU_TIMER)) || (!(flags & CPU_TIMER) && !(flags & GPU_TIMER)) ) // xor
            {
                return 1;
            }

            if ( m_timers.find(std::string(label)) != m_timers.end())
            {
                return 1;
            }

            if (flags & CPU_TIMER)
            {
                timer = new TimerCPU(flags & ACCUMULATE_AVERAGE);
            }
            else if (flags & GPU_TIMER)
            {
                timer = new TimerGPU_Events(flags & ACCUMULATE_AVERAGE);
            }

            if (flags & CREATE_AND_START)
            {
                timer->start();
            }

            m_timers[std::string(label)] = timer;
            return 0;
        }

        int startTimer(const char *label)
        {
            if ( m_timers.find(std::string(label)) != m_timers.end())
            {
                m_timers[std::string(label)]->start();
                return 0;
            }
            else
            {
                return 1;
            }
        }

        double elapsedTimer(const char *label)
        {
            if ( m_timers.find(std::string(label)) != m_timers.end())
            {
                return m_timers[std::string(label)]->elapsed();
            }
            else { return 0.0; }
        }

        double stopTimer(const char *label)
        {
            if ( m_timers.find(std::string(label)) != m_timers.end())
            {
                return m_timers[std::string(label)]->stop();
            }
            else
            {
                return 0.0;
            }
        }

        double getTotalTime(const char *label)
        {
            if ( m_timers.find(std::string(label)) != m_timers.end())
            {
                return m_timers[std::string(label)]->get_accumulated_time();
            }
            else
            {
                return 0.0;
            }
        }
};

TimerMap &getTimers();

} // end namespace amgx

