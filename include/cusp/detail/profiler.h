/*
 *  Copyright 2008-2010 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*
 * Adapted from Andrew's "High-Performance C++ Profiler" (MIT License)
 *   http://code.google.com/p/high-performance-cplusplus-profiler/
 *
 * Additional details:
 *   http://floodyberry.wordpress.com/2009/10/07/high-performance-cplusplus-profiling/
 *
 */

#pragma once

#define __PROFILER_ENABLED__
#define __PROFILER_FULL_TYPE_EXPANSION__

#undef noinline
#undef fastcall

#if defined(_MSC_VER)
        #undef __PRETTY_FUNCTION__
        #define __PRETTY_FUNCTION__ __FUNCSIG__
        #define PROFILE_CONCAT( a, b ) a "/" b

        #define noinline __declspec(noinline)
        #define fastcall __fastcall
#else
        #define PROFILE_CONCAT( a, b ) b

        #define noinline __attribute__ ((noinline))
        #define fastcall
#endif

#if defined(__PROFILER_FULL_TYPE_EXPANSION__)
        #define PROFILE_FUNCTION() __PRETTY_FUNCTION__
#else
        #define PROFILE_FUNCTION() __FUNCTION__
#endif

#if defined(__PROFILER_ENABLED__)
        // function
        #define PROFILE_PAUSE()             cusp::detail::profiler::pause()
        #define PROFILE_UNPAUSE()           cusp::detail::profiler::unpause()
        #define PROFILE_PAUSE_SCOPED()      cusp::detail::profiler::ScopedPause profilerpause##__LINE__

        #define PROFILE_START_RAW( text )   cusp::detail::profiler::enter( text )
        #define PROFILE_START()             PROFILE_START_RAW( PROFILE_FUNCTION()  )
        #define PROFILE_START_DESC( desc )  PROFILE_START_RAW( PROFILE_CONCAT( PROFILE_FUNCTION(), desc ) )

        #define PROFILE_SCOPED_RAW( text )  cusp::detail::profiler::Scoped profiler##__LINE__ ( text )
        #define PROFILE_SCOPED()            PROFILE_SCOPED_RAW( PROFILE_FUNCTION() )
        #define PROFILE_SCOPED_DESC( desc ) PROFILE_SCOPED_RAW( PROFILE_CONCAT( PROFILE_FUNCTION(), desc ) )

        #define PROFILE_STOP()              profiler::exit()
#else
        #define PROFILE_PAUSE()
        #define PROFILE_UNPAUSE()
        #define PROFILE_PAUSE_SCOPED()

        #define PROFILE_START_RAW( text )
        #define PROFILE_START()
        #define PROFILE_START_DESC( desc )

        #define PROFILE_SCOPED_RAW( text )
        #define PROFILE_SCOPED()
        #define PROFILE_SCOPED_DESC( desc )

        #define PROFILE_STOP()
#endif

namespace cusp
{
namespace detail
{
namespace profiler
{
        /*
        =============
        Interface functions
        =============
        */

        void detect( int argc, const char *argv[] );
        void detect( const char *commandLine );
        void dump();
        void fastcall enter( const char *name );
        void fastcall exit();
        void fastcall pause();
        void fastcall unpause();
        void reset();

        struct Scoped 
	{
                Scoped( const char *name ) { PROFILE_START_RAW( name ); }
                ~Scoped() { PROFILE_STOP(); }
        };

        struct ScopedPause 
	{
                ScopedPause() { PROFILE_PAUSE(); }
                ~ScopedPause() { PROFILE_UNPAUSE(); }
        };

} // end namespace profiler
} // end namespace detail
} // end namespace cusp

#include <cusp/detail/profiler.inl>

