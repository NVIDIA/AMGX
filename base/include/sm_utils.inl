/* Copyright (c) 2013-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "amgx_types/util.h"
#include "amgx_types/math.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////   Device-level generalized utils, should be included only in files compiled by nvcc  ////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __PTR   "l"
#else
#define __PTR   "r"
#endif

namespace utils
{

#if 1
#define DEFAULT_MASK 0xffffffff
#else
// cannot use as default parameter to functions
#define DEFAULT_MASK activemask()
#endif

// ====================================================================================================================
// Atomics.
// ====================================================================================================================

static __device__ __forceinline__ void atomic_add( float *address, float value )
{
    atomicAdd( address, value );
}

static __device__ __forceinline__ void atomic_add( double *address, double value )
{
#if __CUDA_ARCH__ >= 600
    atomicAdd( address, value );
#else
    unsigned long long *address_as_ull = (unsigned long long *) address;
    unsigned long long old = __double_as_longlong( address[0] ), assumed;

    do
    {
        assumed = old;
        old = atomicCAS( address_as_ull, assumed, __double_as_longlong( value + __longlong_as_double( assumed ) ) );
    }
    while ( assumed != old );

#endif
}

static __device__ __forceinline__ void atomic_add( cuComplex *address, cuComplex value )
{
    atomicAdd((float *)(address), cuCrealf(value));
    atomicAdd((float *)((char *)(address) + sizeof(float)), cuCimagf(value));
}

static __device__ __forceinline__ void atomic_add( cuDoubleComplex *address, cuDoubleComplex value )
{
    atomic_add((double *)(address), cuCreal(value));
    atomic_add((double *)((char *)(address) + sizeof(double)), cuCimag(value));
}

// ====================================================================================================================
// Bit tools.
// ====================================================================================================================

static __device__ __forceinline__ int bfe( int src, int num_bits )
{
    unsigned mask;
    asm( "bfe.u32 %0, %1, 0, %2;" : "=r"(mask) : "r"(src), "r"(num_bits) );
    return mask;
}

static __device__ __forceinline__ int bfind( int src )
{
    int msb;
    asm( "bfind.u32 %0, %1;" : "=r"(msb) : "r"(src) );
    return msb;
}

static __device__ __forceinline__ int bfind( unsigned long long src )
{
    int msb;
    asm( "bfind.u64 %0, %1;" : "=r"(msb) : "l"(src) );
    return msb;
}

static __device__ __forceinline__ unsigned long long brev( unsigned long long src )
{
    unsigned long long rev;
    asm( "brev.b64 %0, %1;" : "=l"(rev) : "l"(src) );
    return rev;
}

// ====================================================================================================================
// Warp tools.
// ====================================================================================================================

static __device__ __forceinline__ int lane_id()
{
    int id;
    asm( "mov.u32 %0, %%laneid;" : "=r"(id) );
    return id;
}

static __device__ __forceinline__ int lane_mask_lt()
{
    int mask;
    asm( "mov.u32 %0, %%lanemask_lt;" : "=r"(mask) );
    return mask;
}

static __device__ __forceinline__ int warp_id()
{
    return threadIdx.x >> 5;
}

// ====================================================================================================================
// Loads.
// ====================================================================================================================

enum Ld_mode { LD_AUTO = 0, LD_CA, LD_CG, LD_TEX, LD_NC };

template< Ld_mode Mode >
struct Ld {};

template<>
struct Ld<LD_AUTO>
{
    template< typename T >
    static __device__ __forceinline__ T load( const T *ptr ) { return *ptr; }
};

template<>
struct Ld<LD_CG>
{
    static __device__ __forceinline__ int load( const int *ptr )
    {
        int ret;
        asm volatile ( "ld.global.cg.s32 %0, [%1];"  : "=r"(ret) : __PTR(ptr) );
        return ret;
    }

    static __device__ __forceinline__ float load( const float *ptr )
    {
        float ret;
        asm volatile ( "ld.global.cg.f32 %0, [%1];"  : "=f"(ret) : __PTR(ptr) );
        return ret;
    }

    static __device__ __forceinline__ double load( const double *ptr )
    {
        double ret;
        asm volatile ( "ld.global.cg.f64 %0, [%1];"  : "=d"(ret) : __PTR(ptr) );
        return ret;
    }

    static __device__ __forceinline__ cuComplex load( const cuComplex *ptr )
    {
        float ret[2];
        asm volatile ( "ld.global.cg.v2.f32 {%0, %1}, [%2];"  : "=f"(ret[0]), "=f"(ret[1]) : __PTR( (float *)(ptr) ) );
        return make_cuComplex(ret[0], ret[1]);
    }

    static __device__ __forceinline__ cuDoubleComplex load( const cuDoubleComplex *ptr )
    {
        double ret[2];
        asm volatile ( "ld.global.cg.v2.f64 {%0, %1}, [%2];"  : "=d"(ret[0]), "=d"(ret[1]) : __PTR( (double *)(ptr) ) );
        return make_cuDoubleComplex(ret[0], ret[1]);
    }

};

template<>
struct Ld<LD_CA>
{
    static __device__ __forceinline__ int load( const int *ptr )
    {
        int ret;
        asm volatile ( "ld.global.ca.s32 %0, [%1];"  : "=r"(ret) : __PTR(ptr) );
        return ret;
    }

    static __device__ __forceinline__ float load( const float *ptr )
    {
        float ret;
        asm volatile ( "ld.global.ca.f32 %0, [%1];"  : "=f"(ret) : __PTR(ptr) );
        return ret;
    }

    static __device__ __forceinline__ double load( const double *ptr )
    {
        double ret;
        asm volatile ( "ld.global.ca.f64 %0, [%1];"  : "=d"(ret) : __PTR(ptr) );
        return ret;
    }

    static __device__ __forceinline__ cuComplex load( const cuComplex *ptr )
    {
        float ret[2];
        asm volatile ( "ld.global.ca.v2.f32 {%0, %1}, [%2];"  : "=f"(ret[0]), "=f"(ret[1]) : __PTR( (float *)(ptr) ) );
        return make_cuComplex(ret[0], ret[1]);
    }

    static __device__ __forceinline__ cuDoubleComplex load( const cuDoubleComplex *ptr )
    {
        double ret[2];
        asm volatile ( "ld.global.ca.v2.f64 {%0, %1}, [%2];"  : "=d"(ret[0]), "=d"(ret[1]) : __PTR( (double *)(ptr) ) );
        return make_cuDoubleComplex(ret[0], ret[1]);
    }
};

template<>
struct Ld<LD_NC>
{
    template< typename T >
    static __device__ __forceinline__ T load( const T *ptr ) { return __ldg( ptr ); }
};


// ====================================================================================================================
// Vector loads.
// ====================================================================================================================

static __device__ __forceinline__ void load_vec2( float (&u)[2], const float *ptr )
{
    asm( "ld.global.cg.v2.f32 {%0, %1}, [%2];" : "=f"(u[0]), "=f"(u[1]) : __PTR(ptr) );
}

static __device__ __forceinline__ void load_vec2( double (&u)[2], const double *ptr )
{
    asm( "ld.global.cg.v2.f64 {%0, %1}, [%2];" : "=d"(u[0]), "=d"(u[1]) : __PTR(ptr) );
}

static __device__ __forceinline__ void load_vec4( float (&u)[4], const float *ptr )
{
    asm( "ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(u[0]), "=f"(u[1]), "=f"(u[2]), "=f"(u[3]) : __PTR(ptr) );
}

static __device__ __forceinline__ void load_vec4( double (&u)[4], const double *ptr )
{
    asm( "ld.global.cg.v2.f64 {%0, %1}, [%2];" : "=d"(u[0]), "=d"(u[1]) : __PTR(ptr + 0) );
    asm( "ld.global.cg.v2.f64 {%0, %1}, [%2];" : "=d"(u[2]), "=d"(u[3]) : __PTR(ptr + 2) );
}

// ====================================================================================================================
// Warp vote functions
// ====================================================================================================================
static __device__ __forceinline__ unsigned int ballot(int p, unsigned int mask = DEFAULT_MASK)
{
#if CUDART_VERSION >= 9000
    return __ballot_sync(mask, p);
#else
    return __ballot(p);   
#endif
}

static __device__ __forceinline__ unsigned int any(int p, unsigned int mask = DEFAULT_MASK)
{
#if CUDART_VERSION >= 9000
    return __any_sync(mask, p);
#else
    return __any(p);   
#endif
}

static __device__ __forceinline__ unsigned int all(int p, unsigned int mask = DEFAULT_MASK)
{
#if CUDART_VERSION >= 9000
    return __all_sync(mask, p);
#else
    return __all(p);   
#endif
}

static __device__ __forceinline__ unsigned int activemask()
{
#if CUDART_VERSION >= 9000
    return __activemask();
#else
    return 0xffffffff;
#endif
}

static __device__ __forceinline__ void syncwarp(unsigned int mask = 0xffffffff)
{
#if CUDART_VERSION >= 9000
    return __syncwarp(mask);
#else
    return;
#endif
}

// ====================================================================================================================
// Shuffle.
// ====================================================================================================================
static __device__ __forceinline__ int shfl( int r, int lane, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
#if CUDART_VERSION >= 9000
    return __shfl_sync( mask, r, lane, bound );
#else
    return __shfl( r, lane, bound );
#endif
}


static __device__ __forceinline__ float shfl( float r, int lane, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
#if CUDART_VERSION >= 9000
    return __shfl_sync( mask, r, lane, bound );
#else
    return __shfl( r, lane, bound );
#endif
}

static __device__ __forceinline__ double shfl( double r, int lane, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
#if CUDART_VERSION >= 9000
    return __shfl_sync(mask, r, lane, bound );
#else
    int hi = __shfl( __double2hiint(r), lane, bound );
    int lo = __shfl( __double2loint(r), lane, bound );
    return __hiloint2double( hi, lo );
#endif
}

static __device__ __forceinline__ cuComplex shfl( cuComplex r, int lane, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
#if CUDART_VERSION >= 9000
    float re = __shfl_sync( mask, cuCrealf(r), lane, bound );
    float im = __shfl_sync( mask, cuCimagf(r), lane, bound );
    return make_cuComplex(re, im);
#else
    float re = __shfl( cuCrealf(r), lane, bound );
    float im = __shfl( cuCimagf(r), lane, bound );
    return make_cuComplex(re, im);
#endif
}

static __device__ __forceinline__ cuDoubleComplex shfl( cuDoubleComplex r, int lane, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
    double re = shfl( cuCreal(r), lane, mask, bound );
    double im = shfl( cuCimag(r), lane, mask, bound );
    return make_cuDoubleComplex( re, im );
}

static __device__ __forceinline__ int shfl_xor( int r, int lane_mask, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
#if CUDART_VERSION >= 9000
    return __shfl_xor_sync( mask, r, lane_mask, bound );
#else
    return __shfl_xor( r, lane_mask, bound );
#endif
}


static __device__ __forceinline__ float shfl_xor( float r, int lane_mask, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
#if CUDART_VERSION >= 9000
    return __shfl_xor_sync( mask, r, lane_mask, bound );
#else
    return __shfl_xor( r, lane_mask, bound );
#endif
}

static __device__ __forceinline__ double shfl_xor( double r, int lane_mask, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
#if CUDART_VERSION >= 9000
    return __shfl_xor_sync( mask, r, lane_mask, bound );
#else
    int hi = __shfl_xor( __double2hiint(r), lane_mask, bound );
    int lo = __shfl_xor( __double2loint(r), lane_mask, bound );
    return __hiloint2double( hi, lo );
#endif
}

static __device__ __forceinline__ cuComplex shfl_xor( cuComplex r, int lane_mask, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
#if CUDART_VERSION >= 9000
    float re = __shfl_xor_sync( mask, cuCrealf(r), lane_mask, bound );
    float im = __shfl_xor_sync( mask, cuCimagf(r), lane_mask, bound );
    return make_cuComplex(re, im);
#else
    float re = __shfl_xor( cuCrealf(r), lane_mask, bound );
    float im = __shfl_xor( cuCimagf(r), lane_mask, bound );
    return make_cuComplex(re, im);
#endif
}

static __device__ __forceinline__ cuDoubleComplex shfl_xor( cuDoubleComplex r, int lane_mask, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
    double re = shfl_xor( cuCreal(r), lane_mask, mask, bound );
    double im = shfl_xor( cuCimag(r), lane_mask, mask, bound );
    return make_cuDoubleComplex( re, im );
}

static __device__ __forceinline__ int shfl_down( int r, int offset, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
#if CUDART_VERSION >= 9000
    return __shfl_down_sync( mask, r, offset, bound );
#else
    return __shfl_down( r, offset, bound );
#endif
}

static __device__ __forceinline__ float shfl_down( float r, int offset, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
#if CUDART_VERSION >= 9000
    return __shfl_down_sync( mask, r, offset, bound );
#else
    return __shfl_down( r, offset, bound );
#endif
}

static __device__ __forceinline__ double shfl_down( double r, int offset, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
#if CUDART_VERSION >= 9000
    return __shfl_down_sync( mask, r, offset, bound );
#else
    int hi = __shfl_down( __double2hiint(r), offset, bound );
    int lo = __shfl_down( __double2loint(r), offset, bound );
    return __hiloint2double( hi, lo );
#endif
}

static __device__ __forceinline__ cuComplex shfl_down( cuComplex r, int lane, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
#if CUDART_VERSION >= 9000
    float re = __shfl_down_sync( mask, cuCrealf(r), lane, bound );
    float im = __shfl_down_sync( mask, cuCimagf(r), lane, bound );
    return make_cuComplex(re, im);
#else
    float re = __shfl_down( cuCrealf(r), lane, bound );
    float im = __shfl_down( cuCimagf(r), lane, bound );
    return make_cuComplex(re, im);
#endif
}

static __device__ __forceinline__ cuDoubleComplex shfl_down( cuDoubleComplex r, int lane, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
    double re = shfl_down( cuCreal(r), lane, bound );
    double im = shfl_down( cuCimag(r), lane, bound );
    return make_cuDoubleComplex( re, im );
}


static __device__ __forceinline__ int shfl_up( int r, int offset, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
#if CUDART_VERSION >= 9000
    return __shfl_up_sync( mask, r, offset, bound );
#else
    return __shfl_up( r, offset, bound );
#endif
}

static __device__ __forceinline__ float shfl_up( float r, int offset, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
#if CUDART_VERSION >= 9000
    return __shfl_up_sync( mask, r, offset, bound );
#else
    return __shfl_up( r, offset, bound );
#endif
}

static __device__ __forceinline__ double shfl_up( double r, int offset, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
#if CUDART_VERSION >= 9000
    return __shfl_up_sync( mask, r, offset, bound );
#else
    int hi = __shfl_up( __double2hiint(r), offset, bound );
    int lo = __shfl_up( __double2loint(r), offset, bound );
    return __hiloint2double( hi, lo );
#endif
}

static __device__ __forceinline__ cuComplex shfl_up( cuComplex r, int lane, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
#if CUDART_VERSION >= 9000
    float re = __shfl_up_sync( mask, cuCrealf(r), lane, bound );
    float im = __shfl_up_sync( mask, cuCimagf(r), lane, bound );
    return make_cuComplex(re, im);
#else
    float re = __shfl_up( cuCrealf(r), lane, bound );
    float im = __shfl_up( cuCimagf(r), lane, bound );
    return make_cuComplex(re, im);
#endif
}

static __device__ __forceinline__ cuDoubleComplex shfl_up( cuDoubleComplex r, int lane, int bound = warpSize, unsigned int mask = DEFAULT_MASK )
{
    double re = shfl_up( cuCreal(r), lane, bound );
    double im = shfl_up( cuCimag(r), lane, bound );
    return make_cuDoubleComplex( re, im );
}

// ====================================================================================================================
// Warp-level reductions.
// ====================================================================================================================

struct Add
{
    template< typename Value_type >
    static __device__ __forceinline__ Value_type eval( Value_type x, Value_type y ) { return x + y; }
};

template< int NUM_THREADS_PER_ITEM, int WarpSize >
struct Warp_reduce_pow2
{
    template< typename Operator, typename Value_type >
    static __device__ __inline__ Value_type execute( Value_type x )
    {
#pragma unroll

        for ( int mask = WarpSize / 2 ; mask >= NUM_THREADS_PER_ITEM ; mask >>= 1 )
        {
            x = Operator::eval( x, shfl_xor(x, mask) );
        }

        return x;
    }
};

template< int NUM_THREADS_PER_ITEM, int WarpSize >
struct Warp_reduce_linear
{
    template< typename Operator, typename Value_type >
    static __device__ __inline__ Value_type execute( Value_type x )
    {
        const int NUM_STEPS = WarpSize / NUM_THREADS_PER_ITEM;
        int my_lane_id = utils::lane_id();
#pragma unroll

        for ( int i = 1 ; i < NUM_STEPS ; ++i )
        {
            Value_type y = shfl_down( x, i * NUM_THREADS_PER_ITEM );

            if ( my_lane_id < NUM_THREADS_PER_ITEM )
            {
                x = Operator::eval( x, y );
            }
        }

        return x;
    }
};

// ====================================================================================================================

template< int NUM_THREADS_PER_ITEM, int WarpSize = 32 >
struct Warp_reduce : public Warp_reduce_pow2<NUM_THREADS_PER_ITEM, WarpSize> {};

template< int WarpSize >
struct Warp_reduce< 3, WarpSize> : public Warp_reduce_linear< 3, WarpSize> {};

template< int WarpSize >
struct Warp_reduce< 4, WarpSize> : public Warp_reduce_linear< 4, WarpSize> {};

template< int WarpSize >
struct Warp_reduce< 5, WarpSize> : public Warp_reduce_linear< 5, WarpSize> {};

template< int WarpSize >
struct Warp_reduce< 6, WarpSize> : public Warp_reduce_linear< 6, WarpSize> {};

template< int WarpSize >
struct Warp_reduce< 7, WarpSize> : public Warp_reduce_linear< 7, WarpSize> {};

template< int WarpSize >
struct Warp_reduce< 9, WarpSize> : public Warp_reduce_linear< 9, WarpSize> {};

template< int WarpSize >
struct Warp_reduce<10, WarpSize> : public Warp_reduce_linear<10, WarpSize> {};

template< int WarpSize >
struct Warp_reduce<11, WarpSize> : public Warp_reduce_linear<11, WarpSize> {};

template< int WarpSize >
struct Warp_reduce<12, WarpSize> : public Warp_reduce_linear<12, WarpSize> {};

template< int WarpSize >
struct Warp_reduce<13, WarpSize> : public Warp_reduce_linear<13, WarpSize> {};

template< int WarpSize >
struct Warp_reduce<14, WarpSize> : public Warp_reduce_linear<14, WarpSize> {};

template< int WarpSize >
struct Warp_reduce<15, WarpSize> : public Warp_reduce_linear<15, WarpSize> {};

// ====================================================================================================================

template< int NUM_THREADS_PER_ITEM, typename Operator, typename Value_type >
static __device__ __forceinline__ Value_type warp_reduce( Value_type x )
{
    return Warp_reduce<NUM_THREADS_PER_ITEM>::template execute<Operator>( x );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace utils

