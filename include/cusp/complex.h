// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*
 * Copyright (c) 2010, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from U.S. Dept. of Energy) All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the
 * following conditions are met:
 * 
 *     * Redistributions of source code must retain the above
 * copyright notice, this list of conditions and the following
 * disclaimer.
 * 
 *     * Redistributions in binary form must reproduce the
 * above copyright notice, this list of conditions and the
 * following disclaimer in the documentation and/or other
 * materials provided with the distribution.
 * 
 *     * Neither the name of the University of California,
 * Berkeley, nor the names of its contributors may be used to
 * endorse or promote products derived from this software
 * without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * 
*/

/*! \file complex.h
 *  \brief Complex numbers
 */

#pragma once

#include <cusp/detail/config.h>

#ifdef _WIN32
#define _USE_MATH_DEFINES 1  // make sure M_PI is defined
#endif

#include <math.h>
#include <complex>
#include <cuComplex.h>
#include <sstream>
#include <cusp/cmath.h>

namespace cusp
{

  template <typename ValueType> struct complex;
  template <> struct complex<float>;
  template <> struct complex<double>;


  ///  Returns the magnitude of z.
  template<typename ValueType> __host__ __device__ ValueType abs(const complex<ValueType>& z);
  ///  Returns the phase angle of z.
  template<typename ValueType> __host__ __device__ ValueType arg(const complex<ValueType>& z);
  ///  Returns the magnitude of z squared.
  template<typename ValueType> __host__ __device__ ValueType norm(const complex<ValueType>& z);

  ///  Returns the complex conjugate of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> conj(const complex<ValueType>& z);
  ///  Returns the complex with magnitude m and angle theta in radians.
  template<typename ValueType> __host__ __device__ complex<ValueType> polar(const ValueType& m, const ValueType& theta = 0);

  // Arithmetic operators:
  // Multiplication
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator*(const complex<ValueType>& lhs, const complex<ValueType>& rhs);
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator*(const complex<ValueType>& lhs, const ValueType & rhs);
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator*(const ValueType& lhs, const complex<ValueType>& rhs);
  // Division
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator/(const complex<ValueType>& lhs, const complex<ValueType>& rhs);
  template <>
    __host__ __device__
    inline complex<float> operator/(const complex<float>& lhs, const complex<float>& rhs);
  template <>
    __host__ __device__
    inline complex<double> operator/(const complex<double>& lhs, const complex<double>& rhs);

  // Addition
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator+(const complex<ValueType>& lhs, const complex<ValueType>& rhs);
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator+(const complex<ValueType>& lhs, const ValueType & rhs);
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator+(const ValueType& lhs, const complex<ValueType>& rhs);
  // Subtraction
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator-(const complex<ValueType>& lhs, const complex<ValueType>& rhs);
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator-(const complex<ValueType>& lhs, const ValueType & rhs);
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator-(const ValueType& lhs, const complex<ValueType>& rhs);

  // Unary plus and minus
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator+(const complex<ValueType>& rhs);
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator-(const complex<ValueType>& rhs);

  // Transcendentals:
  // Returns the complex cosine of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> cos(const complex<ValueType>& z);
  // Returns the complex hyperbolic cosine of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> cosh(const complex<ValueType>& z);
  // Returns the complex base e exponential of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> exp(const complex<ValueType>& z);
  // Returns the complex natural logarithm of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> log(const complex<ValueType>& z);
  // Returns the complex base 10 logarithm of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> log10(const complex<ValueType>& z);
  // Returns z to the n'th power.
  template<typename ValueType> __host__ __device__ complex<ValueType> pow(const complex<ValueType>& z, const int& n);
  // Returns z to the x'th power.
  template<typename ValueType> __host__ __device__ complex<ValueType> pow(const complex<ValueType>&z, const ValueType&x);
  // Returns z to the z2'th power.
  template<typename ValueType> __host__ __device__ complex<ValueType> pow(const complex<ValueType>&z, 
							  const complex<ValueType>&z2);
  // Returns x to the z'th power.
  template<typename ValueType> __host__ __device__ complex<ValueType> pow(const ValueType& x, const complex<ValueType>& z);
  // Returns the complex sine of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> sin(const complex<ValueType>&z);
  // Returns the complex hyperbolic sine of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> sinh(const complex<ValueType>&z);
  // Returns the complex square root of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> sqrt(const complex<ValueType>&z);
  // Returns the complex tangent of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> tan(const complex<ValueType>&z);
  // Returns the complex hyperbolic tangent of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> tanh(const complex<ValueType>&z);


  // Inverse Trigonometric:
  // Returns the complex arc cosine of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> acos(const complex<ValueType>& z);
  // Returns the complex arc sine of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> asin(const complex<ValueType>& z);
  // Returns the complex arc tangent of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> atan(const complex<ValueType>& z);
  // Returns the complex hyperbolic arc cosine of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> acosh(const complex<ValueType>& z);
  // Returns the complex hyperbolic arc sine of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> asinh(const complex<ValueType>& z);
  // Returns the complex hyperbolic arc tangent of z.
  template<typename ValueType> __host__ __device__ complex<ValueType> atanh(const complex<ValueType>& z);



  // Stream operators:
  template<typename ValueType,class charT, class traits>
    std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os, const complex<ValueType>& z);
  template<typename ValueType, typename charT, class traits>
    std::basic_istream<charT, traits>&
    operator>>(std::basic_istream<charT, traits>& is, complex<ValueType>& z);
  

  // Stream operators
  template<typename ValueType,class charT, class traits>
    std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& os, const complex<ValueType>& z)
    {
      os << '(' << z.real() << ',' << z.imag() << ')';
      return os;
    };

  template<typename ValueType, typename charT, class traits>
    std::basic_istream<charT, traits>&
    operator>>(std::basic_istream<charT, traits>& is, complex<ValueType>& z)
    {
      ValueType re, im;

      charT ch;
      is >> ch;

      if(ch == '(')
      {
        is >> re >> ch;
        if (ch == ',')
        {
          is >> im >> ch;
          if (ch == ')')
          {
            z = complex<ValueType>(re, im);
          }
          else
          {
            is.setstate(std::ios_base::failbit);
          }
        }
        else if (ch == ')')
        {
          z = re;
        }
        else
        {
          is.setstate(std::ios_base::failbit);
        }
      }
      else
      {
        is.putback(ch);
        is >> re;
        z = re;
      }
      return is;
    }

template <typename T>
  struct norm_type {
    typedef T type; 
  };
 template <typename T>
   struct norm_type< complex<T> > { 
   typedef T type;
 };
  
template <typename ValueType>
struct complex
{
public:
  typedef ValueType value_type;

  // Constructors
  __host__ __device__      
    inline complex<ValueType>(const ValueType & re = ValueType(), const ValueType& im = ValueType())
    {
      real(re);
      imag(im);
    }  

  template <class X> 
    __host__ __device__
    inline complex<ValueType>(const complex<X> & z)
    {
      real(z.real());
      imag(z.imag());
    }  
  
  template <class X> 
    __host__ __device__
    inline complex<ValueType>(const std::complex<X> & z)
    {
      real(z.real());
      imag(z.imag());
    }  

  template <typename T>
    __host__ __device__
    inline complex<ValueType>& operator=(const complex<T> z)
    {
      real(z.real());
      imag(z.imag());
      return *this;
    }

  __host__ __device__
    inline complex<ValueType>& operator+=(const complex<ValueType> z)
    {
      real(real()+z.real());
      imag(imag()+z.imag());
      return *this;
    }

  __host__ __device__
    inline complex<ValueType>& operator-=(const complex<ValueType> z)
    {
      real(real()-z.real());
      imag(imag()-z.imag());
      return *this;
    }

  __host__ __device__
    inline complex<ValueType>& operator*=(const complex<ValueType> z)
    {
      *this = *this * z;
      return *this;
    }

  __host__ __device__
    inline complex<ValueType>& operator/=(const complex<ValueType> z)
    {
      *this = *this / z;
      return *this;
    }

  __host__ __device__ inline ValueType real() const volatile;
  __host__ __device__ inline ValueType imag() const volatile;
  __host__ __device__ inline ValueType real() const;
  __host__ __device__ inline ValueType imag() const;
  __host__ __device__ inline void real(ValueType) volatile;
  __host__ __device__ inline void imag(ValueType) volatile;
  __host__ __device__ inline void real(ValueType);
  __host__ __device__ inline void imag(ValueType);
};

// TODO make cuFloatComplex and cuDoubleComplex protected
// TODO see if returning references is a perf hazard

template<>
struct complex <float> : public cuFloatComplex
{
public:
  typedef float value_type;
  __host__ __device__ 
    inline  complex<float>(){};
  __host__ __device__ 
    inline  complex<float>(const float & re, const float& im = float())
    {
      real(re);
      imag(im);
    }  

  // For some reason having the following constructor
  // explicitly makes things faster with at least g++
  __host__ __device__
    complex<float>(const complex<float> & z)
    : cuFloatComplex(z){}

  __host__ __device__
    complex<float>(cuFloatComplex z)
    : cuFloatComplex(z){}
  
  template <class X> 
    inline complex<float>(const std::complex<X> & z)
    {
      real(z.real());
      imag(z.imag());
    }  

  // Member operators
  template <typename T>
    __host__ __device__
    inline volatile complex<float>& operator=(const complex<T> z) volatile
    {
      real(z.real());
      imag(z.imag());
      return *this;
    }

  template <typename T>
    __host__ __device__
    inline complex<float>& operator=(const complex<T> z)
    {
      real(z.real());
      imag(z.imag());
      return *this;
    }

  __host__ __device__ 
    inline complex<float>& operator+=(const complex<float> z)
    {
      real(real()+z.real());
      imag(imag()+z.imag());
      return *this;
    }

  __host__ __device__ 
    inline complex<float>& operator-=(const complex<float> z)
    {
      real(real()-z.real());
      imag(imag()-z.imag());
      return *this;
    }

  __host__ __device__ 
    inline complex<float>& operator*=(const complex<float> z)
    {
      *this = *this * z;
      return *this;
    }

  __host__ __device__ 
    inline complex<float>& operator/=(const complex<float> z)
    {
      *this = *this / z;
      return *this;
    }

  // Let the compiler synthesize the copy and assignment operators.
  __host__ __device__ inline complex<float>(const volatile complex<float> & z)
  {
    real(z.real());
    imag(z.imag());
  }

  __host__ __device__ inline float real() const volatile{ return x; }
  __host__ __device__ inline float imag() const volatile{ return y; }
  __host__ __device__ inline float real() const{ return x; }
  __host__ __device__ inline float imag() const{ return y; }
  __host__ __device__ inline void real(float re)volatile{ x = re; }
  __host__ __device__ inline void imag(float im)volatile{ y = im; }
  __host__ __device__ inline void real(float re){ x = re; }
  __host__ __device__ inline void imag(float im){ y = im; }

  // cast operators
  inline operator std::complex<float>() const { return std::complex<float>(real(),imag()); }
  //  inline operator float() const { return real(); }
};

template<>
struct complex <double> : public cuDoubleComplex
{
public:
  typedef double value_type;
  __host__ __device__
    inline  complex<double>(){};
  __host__ __device__
    inline complex<double>(const double & re, const double& im = double())
    {
      real(re);
      imag(im);
    }

  // For some reason having the following constructor
  // explicitly makes things faster with at least g++
  __host__ __device__
    inline complex<double>(const complex<double> & z)
    : cuDoubleComplex(z) {}

  __host__ __device__
    inline complex<double>(cuDoubleComplex z)
    : cuDoubleComplex(z) {}

  template <class X> 
    inline complex<double>(const std::complex<X> & z)
    {
      real(z.real());
      imag(z.imag());
    }  

  // Member operators
  template <typename T>
    __host__ __device__
    inline volatile complex<double>& operator=(const complex<T> z) volatile
    {
      real(z.real());
      imag(z.imag());
      return *this;
    }

  template <typename T>
    __host__ __device__
    inline complex<double>& operator=(const complex<T> z)
    {
      real(z.real());
      imag(z.imag());
      return *this;
    }

  __host__ __device__
    inline complex<double>& operator+=(const complex<double> z)
    {
      real(real()+z.real());
      imag(imag()+z.imag());
      return *this;
    }

  __host__ __device__
    inline complex<double>& operator-=(const complex<double> z)
    {
      real(real()-z.real());
      imag(imag()-z.imag());
      return *this;
    }

  __host__ __device__
    inline complex<double>& operator*=(const complex<double> z)
    {
      *this = *this * z;
      return *this;
    }

  __host__ __device__
    inline complex<double>& operator/=(const complex<double> z)
    {
      *this = *this / z;
      return *this;
    }

  __host__ __device__ inline complex<double>(const volatile complex<double> & z)
  {
    real(z.real());
    imag(z.imag());
  }

  // Let the compiler synthesize the copy and assignment operators.
  __host__ __device__ inline double real() const volatile { return x; }
  __host__ __device__ inline double imag() const volatile { return y; }
  __host__ __device__ inline double real() const { return x; }
  __host__ __device__ inline double imag() const { return y; }
  __host__ __device__ inline void real(double re)volatile{ x = re; }
  __host__ __device__ inline void imag(double im)volatile{ y = im; }
  __host__ __device__ inline void real(double re){ x = re; }
  __host__ __device__ inline void imag(double im){ y = im; }

  // cast operators
  inline operator std::complex<double>() const { return std::complex<double>(real(),imag()); }
  //  inline operator double() { return real(); }
};



  // Binary arithmetic operations
  // At the moment I'm implementing the basic functions, and the 
  // corresponding cuComplex calls are commented.

  template<typename ValueType>
    __host__ __device__ 
    inline complex<ValueType> operator+(const complex<ValueType>& lhs,
					    const complex<ValueType>& rhs){
    return complex<ValueType>(lhs.real()+rhs.real(),lhs.imag()+rhs.imag());
    //  return cuCaddf(lhs,rhs);
  }

  template<typename ValueType>
    __host__ __device__ 
    inline complex<ValueType> operator+(const volatile complex<ValueType>& lhs,
					    const volatile complex<ValueType>& rhs){
    return complex<ValueType>(lhs.real()+rhs.real(),lhs.imag()+rhs.imag());
    //  return cuCaddf(lhs,rhs);
  }

  template <typename ValueType> 
    __host__ __device__ 
    inline complex<ValueType> operator+(const complex<ValueType>& lhs, const ValueType & rhs){
    return complex<ValueType>(lhs.real()+rhs,lhs.imag());
    //  return cuCaddf(lhs,complex<ValueType>(rhs));
  }
  template <typename ValueType> 
    __host__ __device__ 
    inline complex<ValueType> operator+(const ValueType& lhs, const complex<ValueType>& rhs){
    return complex<ValueType>(rhs.real()+lhs,rhs.imag());
    //  return cuCaddf(complex<float>(lhs),rhs);
  }

  template <typename ValueType> 
    __host__ __device__ 
    inline complex<ValueType> operator-(const complex<ValueType>& lhs, const complex<ValueType>& rhs){
    return complex<ValueType>(lhs.real()-rhs.real(),lhs.imag()-rhs.imag());
    //  return cuCsubf(lhs,rhs);
  }
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator-(const complex<ValueType>& lhs, const ValueType & rhs){
    return complex<ValueType>(lhs.real()-rhs,lhs.imag());
    //  return cuCsubf(lhs,complex<float>(rhs));
  }
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator-(const ValueType& lhs, const complex<ValueType>& rhs){
    return complex<ValueType>(lhs-rhs.real(),-rhs.imag());
    //  return cuCsubf(complex<float>(lhs),rhs);
  }

  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator*(const complex<ValueType>& lhs,
					    const complex<ValueType>& rhs){
    return complex<ValueType>(lhs.real()*rhs.real()-lhs.imag()*rhs.imag(),
				  lhs.real()*rhs.imag()+lhs.imag()*rhs.real());
    //  return cuCmulf(lhs,rhs);
  }

  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator*(const complex<ValueType>& lhs, const ValueType & rhs){
    return complex<ValueType>(lhs.real()*rhs,lhs.imag()*rhs);
    //  return cuCmulf(lhs,complex<float>(rhs));
  }

  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator*(const ValueType& lhs, const complex<ValueType>& rhs){
    return complex<ValueType>(rhs.real()*lhs,rhs.imag()*lhs);
    //  return cuCmulf(complex<float>(lhs),rhs);
  }


  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator/(const complex<ValueType>& lhs, const complex<ValueType>& rhs){
    const ValueType cross_norm  =  lhs.real() * rhs.real() + lhs.imag() * rhs.imag();
    const ValueType rhs_norm = norm(rhs);
    return complex<ValueType>(cross_norm/rhs_norm,
				  (lhs.imag() * rhs.real() - lhs.real() * rhs.imag()) / rhs_norm);
  }

  template <>
    __host__ __device__
    inline complex<float> operator/(const complex<float>& lhs, const complex<float>& rhs){
    return cuCdivf(lhs,rhs);
  }

  template <>
    __host__ __device__
    inline complex<double> operator/(const complex<double>& lhs, const complex<double>& rhs){
    return cuCdiv(lhs,rhs);
  }

  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator/(const complex<ValueType>& lhs, const ValueType & rhs){
    return complex<ValueType>(lhs.real()/rhs,lhs.imag()/rhs);
    //  return cuCdivf(lhs,complex<float>(rhs));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> operator/(const ValueType& lhs, const complex<ValueType>& rhs){
    const ValueType cross_norm  =  lhs * rhs.real();
    const ValueType rhs_norm = norm(rhs);
    return complex<ValueType>(cross_norm/rhs_norm,(-lhs.real() * rhs.imag()) / rhs_norm);
  }

  template <>
    __host__ __device__
    inline complex<float> operator/(const float& lhs, const complex<float>& rhs){
    return cuCdivf(complex<float>(lhs),rhs);
  }
  template <>
    __host__ __device__
    inline complex<double> operator/(const double& lhs, const complex<double>& rhs){
    return cuCdiv(complex<double>(lhs),rhs);
  }


  // Unary arithmetic operations
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator+(const complex<ValueType>& rhs){
    return rhs;
  }
  template <typename ValueType> 
    __host__ __device__
    inline complex<ValueType> operator-(const complex<ValueType>& rhs){
    return rhs*-ValueType(1);
  }

  // Equality operators 
  template <typename ValueType> 
    __host__ __device__
    inline bool operator==(const complex<ValueType>& lhs, const complex<ValueType>& rhs){
    if(lhs.real() == rhs.real() && lhs.imag() == rhs.imag()){
      return true;
    }
    return false;
  }
  template <typename ValueType> 
    __host__ __device__
    inline bool operator==(const ValueType & lhs, const complex<ValueType>& rhs){
    if(lhs == rhs.real() && rhs.imag() == 0){
      return true;
    }
    return false;
  }
  template <typename ValueType> 
    __host__ __device__
    inline bool operator==(const complex<ValueType> & lhs, const ValueType& rhs){
    if(lhs.real() == rhs && lhs.imag() == 0){
      return true;
    }
    return false;
  }

  template <typename ValueType> 
    __host__ __device__
    inline bool operator!=(const complex<ValueType>& lhs, const complex<ValueType>& rhs){
    return !(lhs == rhs);
  }

  template <typename ValueType> 
    __host__ __device__
    inline bool operator!=(const ValueType & lhs, const complex<ValueType>& rhs){
    return !(lhs == rhs);
  }

  template <typename ValueType> 
    __host__ __device__
    inline bool operator!=(const complex<ValueType> & lhs, const ValueType& rhs){
    return !(lhs == rhs);
  }


  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> conj(const complex<ValueType>& z){
    return complex<ValueType>(z.real(),-z.imag());
  }

  template <typename ValueType>
    __host__ __device__
    inline ValueType abs(const complex<ValueType>& z){
    return ::hypot(z.real(),z.imag());
  }
  /*
  template <>
    __host__ __device__
    inline float abs(const complex<float>& z){
    return ::hypotf(z.real(),z.imag());
  }*/
  template<>
    __host__ __device__
    inline double abs(const complex<double>& z){
    return ::hypot(z.real(),z.imag());
  }

  template <typename ValueType>
    __host__ __device__
    inline ValueType arg(const complex<ValueType>& z){
    return atan2(z.imag(),z.real());
  }
  template<>
    __host__ __device__
    inline float arg(const complex<float>& z){
    return atan2f(z.imag(),z.real());
  }
  template<>
    __host__ __device__
    inline double arg(const complex<double>& z){
    return atan2(z.imag(),z.real());
  }

  template <typename ValueType>
    __host__ __device__
    inline ValueType norm(const complex<ValueType>& z){
    return z.real()*z.real() + z.imag()*z.imag();
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> polar(const ValueType & m, const ValueType & theta){ 
    return complex<ValueType>(m * ::cos(theta),m * ::sin(theta));
  }

  template <>
    __host__ __device__
    inline complex<float> polar(const float & magnitude, const float & angle){ 
    return complex<float>(magnitude * ::cosf(angle),magnitude * ::sinf(angle));
  }

  template <>
    __host__ __device__
    inline complex<double> polar(const double & magnitude, const double & angle){ 
    return complex<double>(magnitude * ::cos(angle),magnitude * ::sin(angle));
  }

  // Transcendental functions implementation
  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> cos(const complex<ValueType>& z){
    const ValueType re = z.real();
    const ValueType im = z.imag();
    return complex<ValueType>(::cos(re) * ::cosh(im), -::sin(re) * ::sinh(im));
  }

  template <>
    __host__ __device__
    inline complex<float> cos(const complex<float>& z){
    const float re = z.real();
    const float im = z.imag();
    return complex<float>(cosf(re) * coshf(im), -sinf(re) * sinhf(im));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> cosh(const complex<ValueType>& z){
    const ValueType re = z.real();
    const ValueType im = z.imag();
    return complex<ValueType>(::cosh(re) * ::cos(im), ::sinh(re) * ::sin(im));
  }

  template <>
    __host__ __device__
    inline complex<float> cosh(const complex<float>& z){
    const float re = z.real();
    const float im = z.imag();
    return complex<float>(::coshf(re) * ::cosf(im), ::sinhf(re) * ::sinf(im));
  }


  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> exp(const complex<ValueType>& z){
    return polar(::exp(z.real()),z.imag());
  }

  template <>
    __host__ __device__
    inline complex<float> exp(const complex<float>& z){
    return polar(::expf(z.real()),z.imag());
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> log(const complex<ValueType>& z){
    return complex<ValueType>(::log(abs(z)),arg(z));
  }

  template <>
    __host__ __device__
    inline complex<float> log(const complex<float>& z){
    return complex<float>(::logf(abs(z)),arg(z));
  }


  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> log10(const complex<ValueType>& z){ 
    // Using the explicit literal prevents compile time warnings in
    // devices that don't support doubles 
    return log(z)/ValueType(2.30258509299404568402);
    //    return log(z)/ValueType(::log(10.0));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> pow(const complex<ValueType>& z, const ValueType & exponent){
    return exp(log(z)*exponent);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> pow(const complex<ValueType>& z, const complex<ValueType> & exponent){
    return exp(log(z)*exponent);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> pow(const ValueType & x, const complex<ValueType> & exponent){
    return exp(::log(x)*exponent);
  }

  template <>
    __host__ __device__
    inline complex<float> pow(const float & x, const complex<float> & exponent){
    return exp(::logf(x)*exponent);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> pow(const complex<ValueType>& z,const int & exponent){
    return exp(log(z)*ValueType(exponent));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> sin(const complex<ValueType>& z){
    const ValueType re = z.real();
    const ValueType im = z.imag();
    return complex<ValueType>(::sin(re) * ::cosh(im), ::cos(re) * ::sinh(im));
  }

  template <>
    __host__ __device__
    inline complex<float> sin(const complex<float>& z){
    const float re = z.real();
    const float im = z.imag();
    return complex<float>(::sinf(re) * ::coshf(im), ::cosf(re) * ::sinhf(im));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> sinh(const complex<ValueType>& z){
    const ValueType re = z.real();
    const ValueType im = z.imag();
    return complex<ValueType>(::sinh(re) * ::cos(im), ::cosh(re) * ::sin(im));
  }

  template <>
    __host__ __device__
    inline complex<float> sinh(const complex<float>& z){
    const float re = z.real();
    const float im = z.imag();
    return complex<float>(::sinhf(re) * ::cosf(im), ::coshf(re) * ::sinf(im));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> sqrt(const complex<ValueType>& z){
    return polar(::sqrt(abs(z)),arg(z)/ValueType(2));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<float> sqrt(const complex<float>& z){
    return polar(::sqrtf(abs(z)),arg(z)/float(2));
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> tan(const complex<ValueType>& z){
    return sin(z)/cos(z);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> tanh(const complex<ValueType>& z){
    // This implementation seems better than the simple sin/cos
    return (exp(ValueType(2)*z)-ValueType(1))/(exp(ValueType(2)*z)+ValueType(1));
    //    return sinh(z)/cosh(z);
  }

  // Inverse trigonometric functions implementation

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> acos(const complex<ValueType>& z){
    const complex<ValueType> ret = asin(z);
    return complex<ValueType>(ValueType(M_PI/2.0) - ret.real(),-ret.imag());
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> asin(const complex<ValueType>& z){
    const complex<ValueType> i(0,1);
    return -i*asinh(i*z);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> atan(const complex<ValueType>& z){
    const complex<ValueType> i(0,1);
    return -i*atanh(i*z);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> acosh(const complex<ValueType>& z){
    cusp::complex<ValueType> ret((z.real() - z.imag()) * (z.real() + z.imag()) - ValueType(1.0),
				 ValueType(2.0) * z.real() * z.imag());    
    ret = sqrt(ret);
    if (z.real() < ValueType(0.0)){
      ret = -ret;
    }
    ret += z;
    ret = log(ret);
    if (ret.real() < ValueType(0.0)){
      ret = -ret;
    }
    return ret;

    /*
    cusp::complex<ValueType> ret = log(sqrt(z*z-ValueType(1))+z);
    if(ret.real() < 0){
      ret.real(-ret.real());
    }
    return ret;
    */
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> asinh(const complex<ValueType>& z){
    return log(sqrt(z*z+ValueType(1))+z);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<ValueType> atanh(const complex<ValueType>& z){
    ValueType imag2 = z.imag() *  z.imag();   
    ValueType n = ValueType(1.0) + z.real();
    n = imag2 + n * n;

    ValueType d = ValueType(1.0) - z.real();
    d = imag2 + d * d;
    complex<ValueType> ret(ValueType(0.25) * (::log(n) - ::log(d)),0);

    d = ValueType(1.0) -  z.real() * z.real() - imag2;

    ret.imag(ValueType(0.5) * ::atan2(ValueType(2.0) * z.imag(), d));
    return ret;
    //return (log(ValueType(1)+z)-log(ValueType(1)-z))/ValueType(2);
  }

  template <typename ValueType>
    __host__ __device__
    inline complex<float> atanh(const complex<float>& z){
        float imag2 = z.imag() *  z.imag();   
    float n = float(1.0) + z.real();
    n = imag2 + n * n;

    float d = float(1.0) - z.real();
    d = imag2 + d * d;
    complex<float> ret(float(0.25) * (::logf(n) - ::logf(d)),0);

    d = float(1.0) -  z.real() * z.real() - imag2;

    ret.imag(float(0.5) * ::atan2f(float(2.0) * z.imag(), d));
    return ret;
    //return (log(ValueType(1)+z)-log(ValueType(1)-z))/ValueType(2);

  }

} // end namespace cusp
