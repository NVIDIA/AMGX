// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <operators/operator.h>

namespace amgx
{

template <typename T_Config>
class ShiftedOperator : public Operator<T_Config>
{
    public:
        typedef T_Config TConfig;
        typedef Operator<TConfig> Base;

        typedef typename TConfig::VecPrec ValueTypeVec;
        typedef typename TConfig::IndPrec IndType;

        ShiftedOperator(Operator<TConfig> &A, ValueTypeVec shift)
            : m_A(&A), m_shift(shift)
        {
        }

        ~ShiftedOperator()
        {
        }

        void apply(const Vector<TConfig> &v, Vector<TConfig> &res, ViewType view = OWNED);

        DistributedManager<TConfig> *getManager() const
        {
            return m_A->getManager();
        }

        IndType get_num_rows() const
        {
            return m_A->get_num_rows();
        }
        IndType get_num_cols() const
        {
            return m_A->get_num_cols();
        }
        IndType get_block_dimx() const
        {
            return m_A->get_block_dimx();
        }
        IndType get_block_dimy() const
        {
            return m_A->get_block_dimy();
        }
        IndType get_block_size() const
        {
            return m_A->get_block_size();
        }

        bool is_matrix_singleGPU() const
        {
            return m_A->is_matrix_singleGPU();
        }

        ViewType currentView() const
        {
            return m_A->currentView();
        }
        void setView(ViewType type)
        {
            m_A->setView(type);
        }
        void setViewInterior()
        {
            m_A->setViewInterior();
        }
        void setViewExterior()
        {
            m_A->setViewExterior();
        }
        void setInteriorView(ViewType view)
        {
            m_A->setInteriorView(view);
        }
        void setExteriorView(ViewType view)
        {
            m_A->setExteriorView(view);
        }
        ViewType getViewInterior() const
        {
            return m_A->getViewInterior();
        }
        ViewType getViewExterior() const
        {
            return m_A->getViewExterior();
        }

        void getOffsetAndSizeForView(ViewType type, int *offset, int *size) const
        {
            m_A->getOffsetAndSizeForView(type, offset, size);
        }
    private:
        Operator<TConfig> *m_A;
        ValueTypeVec m_shift;
};

}
