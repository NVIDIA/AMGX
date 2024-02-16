// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <distributed/distributed_manager.h>
#include <vector.h>

namespace amgx
{

template <typename T_Config>
class Operator
{
    public:
        typedef T_Config TConfig;
        typedef typename TConfig::MemSpace memory_space;
        typedef typename TConfig::MatPrec  value_type;
        typedef typename TConfig::IndPrec  index_type;

        Operator()
        {
        }

        virtual ~Operator()
        {
        }

        // Apply the operator on vector v and store the result in vector res.
        // Latency hiding must be handled internally by the concrete
        // operator class and cannot be currently propagated to other
        // operators.
        virtual void apply(const Vector<TConfig> &v, Vector<TConfig> &res, ViewType view = OWNED) = 0;

        virtual DistributedManager<TConfig> *getManager() const = 0;

        virtual index_type get_num_rows() const = 0;
        virtual index_type get_num_cols() const = 0;
        virtual index_type get_block_dimx() const = 0;
        virtual index_type get_block_dimy() const = 0;
        virtual index_type get_block_size() const = 0;

        virtual bool is_matrix_singleGPU() const = 0;
        virtual bool is_matrix_distributed() const = 0;

        virtual ViewType currentView() const = 0;
        virtual void setView(ViewType type) = 0;
        virtual void setViewInterior() = 0;
        virtual void setViewExterior() = 0;
        virtual void setInteriorView(ViewType view) = 0;
        virtual void setExteriorView(ViewType view) = 0;
        virtual ViewType getViewInterior() const = 0;
        virtual ViewType getViewExterior() const = 0;

        virtual void getOffsetAndSizeForView(ViewType type, int *offset, int *size) const = 0;
};

}
