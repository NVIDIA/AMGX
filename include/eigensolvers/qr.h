// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <matrix.h>
#include <stack>

namespace amgx
{

template <typename TConfig>
class HouseholderQR
{
    public:
        typedef Matrix<TConfig> TMatrix;
        typedef Vector<TConfig> TVector;

        typedef typename TConfig::template setMemSpace<AMGX_host  >::Type TConfig_h;
        typedef typename TConfig::template setMemSpace<AMGX_device>::Type TConfig_d;

        typedef Vector<TConfig_h> Vector_h;
        typedef Vector<TConfig_d> Vector_d;

        typedef typename TConfig::VecPrec ValueTypeVec;

        HouseholderQR(TMatrix &A);
        void QR_decomposition(TVector &V);
    private:
        void QR(TVector &V);
        void QR(TVector &V, TVector &R);
        void send_vector(TVector &V, int destination);
        void receive_vector(TVector &V, int source);
        void inverse_phase(TVector &V, TVector &R, int root);
    private:
        TMatrix &m_A;
        Vector_h m_tau;
        TVector m_work;
        bool m_use_R_inverse;
        std::stack<TVector> m_local_comms_stack;
};

}
