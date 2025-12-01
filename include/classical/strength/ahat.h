// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <classical/strength/strength_base.h>

namespace amgx
{

template <class T_Config>
class Strength_Ahat : public Strength_Base<T_Config>
{
        typedef T_Config TConfig;
        typedef typename TConfig::MatPrec ValueType;
        typedef typename TConfig::IndPrec IndexType;
    public:
        Strength_Ahat(AMG_Config &cfg, const std::string &cfg_scope) : Strength_Base<T_Config>(cfg, cfg_scope) {}
        __host__ __device__
        bool strongly_connected(ValueType val, ValueType threshold, ValueType diagonal)
        {
            //return (diagonal<0.0) ?  val>threshold : val<threshold;
            return  -val >= -threshold;
        }
};

template<class T_Config>
class Strength_Ahat_StrengthFactory: public StrengthFactory<T_Config>
{
    public:
        Strength<T_Config> *create(AMG_Config &cfg, const std::string &cfg_scope) { return new Strength_Ahat<T_Config>(cfg, cfg_scope); }
};

} // namespace amgx


