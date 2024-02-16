// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <operators/operator.h>

namespace amgx
{

template <typename TConfig> class Solver;

template <typename T_Config>
class SolveOperator : public Operator<T_Config>
{
    public:
        typedef T_Config TConfig;
        typedef Operator<TConfig> Base;

        typedef typename TConfig::VecPrec ValueTypeVec;
        typedef typename TConfig::IndPrec IndType;

        SolveOperator(Operator<TConfig> &A, Solver<TConfig> &solver)
            : m_A(&A), m_solver(&solver)
        {
        }

        ~SolveOperator();

        Operator<TConfig> &get_operator()
        {
            return *m_A;
        }

        void setup();

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

        bool is_matrix_distributed() const
        {
            return m_A->is_matrix_distributed();
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
        Solver<TConfig> *m_solver;
};

}
