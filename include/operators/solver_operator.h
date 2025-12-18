// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <operators/operator.h>
#include <solvers/solver.h>

namespace amgx
{

template <typename T_Config>
class SolverOperator : public Operator<T_Config>
{
    public:
        typedef T_Config TConfig;
        typedef Operator<TConfig> Base;

        typedef typename TConfig::VecPrec ValueTypeVec;
        typedef typename TConfig::IndPrec IndType;

        SolverOperator(Operator<TConfig> *A, Solver<TConfig> *solver)
            : m_A(A), m_solver(solver)
        {
            m_work.resize(m_A->get_num_cols() * m_A->get_block_dimy());
            m_work.set_block_dimy(this->m_A->get_block_dimy());
            m_work.set_block_dimx(1);
            m_work.dirtybit = 1;
            m_work.tag = 7321;
        }

        ~SolverOperator()
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
        Vector<TConfig> m_work;
};

template <typename TConfig>
void SolverOperator<TConfig>::apply(const Vector<TConfig> &v, Vector<TConfig> &res, ViewType view)
{
    Operator<TConfig> &A = *m_A;
    int offset, size;
    A.getOffsetAndSizeForView(view, &offset, &size);
    // we need to drop const here, it's how it is for our solvers - RHS is NOT const
    Vector<TConfig> &cv = const_cast<Vector<TConfig>&> (v);
    m_solver->solve( cv, m_work, true );
    A.apply(m_work, res, OWNED);
}

}
