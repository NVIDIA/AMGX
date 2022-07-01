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

#pragma once

#include <operators/operator.h>
#include "amgx_types/util.h"

namespace amgx
{

template <typename T_Config>
class DeflatedMultiplyOperator : public Operator<T_Config>
{
    public:
        typedef T_Config TConfig;
        typedef Operator<TConfig> Base;

        typedef typename TConfig::VecPrec ValueTypeVec;
        typedef typename TConfig::IndPrec IndType;

        DeflatedMultiplyOperator(Operator<TConfig> &A)
            : m_A(&A), m_x(NULL), m_mu(types::util<ValueTypeVec>::get_zero()), m_work(NULL)
        {
        }

        ~DeflatedMultiplyOperator()
        {
        }

        void set_x(Vector<TConfig> &x)
        {
            m_x = &x;
        }

        void set_mu(ValueTypeVec mu)
        {
            m_mu = mu;
        }

        void set_workspace(Vector<TConfig> &work)
        {
            m_work = &work;
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
        Vector<TConfig> *m_x;
        Vector<TConfig> *m_work;
        ValueTypeVec m_mu;
};

}
