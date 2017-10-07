/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include <basic_types.h>
#include <stdio.h>
#include <amgx_types/pod_types.h>

// some traits and simple operations for basic AMGX datatypes

namespace amgx
{

namespace types
{

// some utility
static inline __host__ __device__ float get_re( const cuComplex &a)
{
    return cuCrealf(a);
}

static inline __host__ __device__ double get_re( const cuDoubleComplex &a)
{
    return cuCreal(a);
}

static inline __host__ __device__ float get_im( const cuComplex &a)
{
    return cuCimagf(a);
}

static inline __host__ __device__ double get_im( const cuDoubleComplex &a)
{
    return cuCimag(a);
}


template < typename T, typename POD_TYPE = typename PODTypes<T>::type >
struct util;
/*{
    static __host__ __device__ __inline__ bool is_real() {return true;};
    static __host__ __device__ __inline__ bool is_complex() {return false;};

    static __host__ __device__ __inline__ T get_zero();
    static __host__ __device__ __inline__ T get_one();
    static __host__ __device__ __inline__ T get_minus_one();
    //maybe just overload operator== for cuComplex and cuDoubleComplex?
    static __host__ __device__ __inline__ bool is_zero(const T& val);
    static __host__ __device__ __inline__ bool is_equal(const T& val1, const T& val2);

    // new_val = -old_val 
    static __host__ __device__ __inline__ T invert(const T& val);
    // cpmplex conjugate. does nothing for real vals
    static __host__ __device__ __inline__ T conjugate(const T& val);
    // same, but in-place. nut sure if needed, usually it's optimized by compiler
    static __host__ __device__ __inline__ void invert_inplace(T& val);
    static __host__ __device__ __inline__ void conjugate_inplace(T& val);

    static __host__ __device__ __inline__ POD_TYPE abs (const T& val);

    // decast volatile for complex types, because cannot overload assignment operator
    static __host__ __device__ __inline__    T volcast (const volatile T& val);
    static __host__ __device__ __inline__ void volcast (const T& val, volatile T* ret);

    // Intented for explicit conversion from ValueTypeA to ValueTypeB because cannot overload typecast for complex datatypes.
    // Don't use if possible, it is not a good stuff. Is used by old gaussian elimination implementation and matrix writer
    template <typename V>
    static __host__ __device__ __inline__ void to_uptype (const T& val, V& ret);

    static void printf(const char* format, const T& val);
    static void fprintf(FILE* f, const char* fmt, const T& val);
};
*/
template <>
struct util <float,  PODTypes<float>::type >
{
    typedef double uptype;
    typedef float downtype;

    static const bool is_real = true;
    static const bool is_complex = false;

    static __host__ __device__ __inline__ float get_zero() { return 0.f; };
    static __host__ __device__ __inline__ float get_one() { return 1.f; };
    static __host__ __device__ __inline__ float get_minus_one() { return -1.f; };
    // exact comparison, which might result wrong answer in a lot of cases
    static __host__ __device__ __inline__ bool is_zero(const float &val) { return val == get_zero(); };
    static __host__ __device__ __inline__ bool is_equal(const float &val1, const float &val2) { return val1 == val2;} ;

    static __host__ __device__ __inline__ float invert(const float &val) {return -val;};
    static __host__ __device__ __inline__ float conjugate(const float &val) {return val;};
    static __host__ __device__ __inline__ void  invert_inplace(float &val) {val = -val;};
    static __host__ __device__ __inline__ void  conjugate_inplace(float &val) {};

    static __host__ __device__ __inline__ float abs (const float &val)
    {
        return fabs(val);
    }

    template <typename V>
    static __host__ __device__ __inline__ void to_uptype (const float &src, V &dst)
    {
        dst = (V)(src);
    }

    static __host__ __device__ __inline__ float to_downtype (const float &src)
    {
        return src;
    }

    static __host__ __device__ __inline__ float volcast (const volatile    float &val) {return val;}
    static __host__ __device__ __inline__ void  volcast (const float &val, volatile float *ret) {*ret = val;}

    /*template <typename M>
    static __host__ __device__ __inline__ float mul(const float& val, const M& mult)
    {
        static_assert(util<M>::is_real(), "Multiply is supported for real constant only");
        return val*mult;
    }*/

    static void printf(const char *fmt, const float &val) { ::printf(fmt, val); };
    static void fprintf(FILE *f, const char *fmt, const float &val) { ::fprintf(f, fmt, val); };
};

template <>
struct util <double, PODTypes<double>::type>
{
    typedef double uptype;
    typedef float downtype;

    static const bool is_real = true;
    static const bool is_complex = false;

    static __host__ __device__ __inline__ double get_zero() { return 0.; };
    static __host__ __device__ __inline__ double get_one() { return 1.; };
    static __host__ __device__ __inline__ double get_minus_one() { return -1.; };

    static __host__ __device__ __inline__ bool is_zero(const double &val) { return val == get_zero(); };
    static __host__ __device__ __inline__ bool is_equal(const double &val1, double &val2) { return val1 == val2;} ;

    static __host__ __device__ __inline__ double invert(const double &val) {return -val;};
    static __host__ __device__ __inline__ double conjugate(const double &val) {return val;};
    static __host__ __device__ __inline__ void invert_inplace(double &val) {val = -val;};
    static __host__ __device__ __inline__ void conjugate_inplace(double &val) {};

    static __host__ __device__ __inline__ double abs (const double &val)
    {
        return fabs(val);
    }

    template <typename V>
    static __host__ __device__ __inline__ void to_uptype (const float &src, V &dst)
    {
        dst = (V)(src);
    }

    static __host__ __device__ __inline__ float to_downtype (const float &src)
    {
        return (float)src;
    }

    static __host__ __device__ __inline__ double volcast (const volatile   double &val) {return val;}
    static __host__ __device__ __inline__ void   volcast (const double &val, volatile double *ret) {*ret = val;}

    /*
    template <typename M>
    static __host__ __device__ __inline__ double mulf(const double& val, const M& mult)
    {
        static_assert(util<M>::is_real(), "Multiply is supported for real constant only");
        return val*mult;
    }*/

    static void printf(const char *fmt, const double &val) { ::printf(fmt, val); };
    static void fprintf(FILE *f, const char *fmt, const double &val) { ::fprintf(f, fmt, val); };
};

template <>
struct util <cuComplex, PODTypes<cuComplex>::type >
{
    typedef cuDoubleComplex uptype;
    typedef cuComplex downtype;

    static const bool is_real = false;
    static const bool is_complex = true;

    static __host__ __device__ __inline__ cuComplex get_zero() { return make_cuComplex(0.f, 0.f); };
    static __host__ __device__ __inline__ cuComplex get_one() { return make_cuComplex(1.f, 0.f); };
    static __host__ __device__ __inline__ cuComplex get_minus_one() { return make_cuComplex(-1.f, 0.f); };

    static __host__ __device__ __inline__ bool is_zero(const cuComplex &val) { return (cuCrealf(val) == 0.f && cuCimagf(val) == 0.f); };
    static __host__ __device__ __inline__ bool is_equal(const cuComplex &val1, const cuComplex &val2) { return (cuCrealf(val1) == cuCrealf(val2) && cuCimagf(val1) == cuCimagf(val2));} ;

    static __host__ __device__ __inline__ cuComplex invert(const cuComplex &val) {return make_cuComplex(-cuCrealf(val), -cuCimagf(val));};
    static __host__ __device__ __inline__ cuComplex conjugate(const cuComplex &val) {return make_cuComplex(cuCrealf(val), -cuCimagf(val));};
    static __host__ __device__ __inline__ void invert_inplace(cuComplex &val) {val = make_cuComplex(-cuCrealf(val), -cuCimagf(val));};
    static __host__ __device__ __inline__ void conjugate_inplace(cuComplex &val) {val = make_cuComplex(cuCrealf(val), -cuCimagf(val));};

    static __host__ __device__ __inline__ float abs (const cuComplex &val)
    {
        return cuCabsf(val);
    }

    template <typename V>
    static __host__ __device__ __inline__ void to_uptype (const cuComplex &src, V &dst)
    {
        dst = (V)(src);
    }

    static __host__ __device__ __inline__ cuComplex to_downtype (const cuComplex &src)
    {
        return src;
    }

    static __host__ __device__ __inline__ cuComplex volcast (const volatile cuComplex &val)
    {
        return make_cuComplex(val.x, val.y);
    }
    static __host__ __device__ __inline__ void      volcast (const cuComplex &val, volatile cuComplex *ret)
    {
        (*ret).x = cuCrealf(val);
        (*ret).y = cuCimagf(val);
    }

    /*template <typename M>
    static __host__ __device__ __inline__ cuComplex mulf(const cuComplex& val, const M& mult)
    {
        static_assert(util<M>::is_real(), "Multiply is supported for real constant only");
        return make_cuComplex(cuCrealf(val)*mult, cuCimagf(val)*mult);
    }*/

    static void printf(const char *fmt, const cuComplex &val)
    {
        ::printf(fmt, cuCrealf(val));
        ::printf(" ");
        ::printf(fmt, cuCimagf(val));
    };
    static void fprintf(FILE *f, const char *fmt, const cuComplex &val)
    {
        ::fprintf(f, fmt, cuCrealf(val));
        ::fprintf(f, " ");
        ::fprintf(f, fmt, cuCimagf(val));
    };
};

template <>
__host__ __device__ __inline__ void util<cuComplex>::to_uptype<cuDoubleComplex> (const cuComplex &src, cuDoubleComplex &dst)
{
    dst = make_cuDoubleComplex(cuCrealf(src), cuCimagf(src));
}

template <>
struct util <cuDoubleComplex, PODTypes<cuDoubleComplex>::type>
{
    typedef cuDoubleComplex uptype;
    typedef cuComplex downtype;

    static const bool is_real = false;
    static const bool is_complex = true;

    static __host__ __device__ __inline__ cuDoubleComplex get_zero() { return make_cuDoubleComplex(0., 0.); };
    static __host__ __device__ __inline__ cuDoubleComplex get_one() { return make_cuDoubleComplex(1., 0.); };
    static __host__ __device__ __inline__ cuDoubleComplex get_minus_one() { return make_cuDoubleComplex(-1., 0.); };

    static __host__ __device__ __inline__ bool is_zero(const cuDoubleComplex &val) { return (cuCreal(val) == 0.f && cuCimag(val) == 0.f); };
    static __host__ __device__ __inline__ bool is_equal(const cuDoubleComplex &val1, const cuDoubleComplex &val2) { return (cuCreal(val1) == cuCreal(val2) && cuCimag(val1) == cuCimag(val2));} ;

    static __host__ __device__ __inline__ cuDoubleComplex invert(const cuDoubleComplex &val) {return make_cuDoubleComplex(-cuCreal(val), -cuCimag(val));};
    static __host__ __device__ __inline__ cuDoubleComplex conjugate(const cuDoubleComplex &val) {return make_cuDoubleComplex(cuCreal(val), -cuCimag(val));};
    static __host__ __device__ __inline__ void invert_inplace(cuDoubleComplex &val) {val =  make_cuDoubleComplex(-cuCreal(val), -cuCimag(val));};
    static __host__ __device__ __inline__ void conjugate_inplace(cuDoubleComplex &val) {val = make_cuDoubleComplex(cuCreal(val), -cuCimag(val));};

    static __host__ __device__ __inline__ double abs (const cuDoubleComplex &val)
    {
        return cuCabs(val);
    }

    template <typename V>
    static __host__ __device__ __inline__ void to_uptype (const cuDoubleComplex &src, V &dst)
    {
        dst = (V)(src);
    }

    static __host__ __device__ __inline__ cuComplex to_downtype (const cuDoubleComplex &src)
    {
        return make_cuComplex(cuCreal(src), cuCimag(src));;
    }

    static __host__ __device__ __inline__ cuDoubleComplex volcast (const volatile cuDoubleComplex &val)
    {
        return make_cuDoubleComplex(val.x, val.y);
    }
    static __host__ __device__ __inline__ void            volcast (const cuDoubleComplex &val, volatile cuDoubleComplex *ret)
    {
        (*ret).x = cuCreal(val);
        (*ret).y = cuCimag(val);
    }

    /*template <typename M>
    static __host__ __device__ __inline__ cuDoubleComplex mulf(const cuDoubleComplex& val, const M& mult)
    {
        static_assert(util<M>::is_real(), "Multiply is supported for real constant only");
        return make_cuDoubleComplex(cuCreal(val)*mult, cuCimag(val)*mult);
    }*/

    static void printf(const char *fmt, const cuDoubleComplex &val)
    {
        ::printf(fmt, cuCreal(val));
        ::printf(" ");
        ::printf(fmt, cuCimag(val));
    };
    static void fprintf(FILE *f, const char *fmt, const cuDoubleComplex &val)
    {
        ::fprintf(f, fmt, cuCreal(val));
        ::fprintf(f, " ");
        ::fprintf(f, fmt, cuCimag(val));
    };
};

} //namespace types

} // namespace amgx
