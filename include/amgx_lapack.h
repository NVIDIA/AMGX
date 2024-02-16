// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector.h>
#include <sstream>

namespace amgx
{

template <class TConfig> class Lapack;

// Specialization for host.
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Lapack< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    private:
        Lapack();
        ~Lapack();
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef typename TConfig::template setMemSpace<AMGX_device>::Type TConfig_d;

        static void not_implemented();
        static void check_lapack_enabled();
        static void check_magma_enabled();

        static void geev(const Vector<TConfig> &A, Vector<TConfig> &eigenvalues);
        static void geev(const Vector<TConfig> &A, Vector<TConfig> &eigenvalues, Vector<TConfig> &eigenvectors);

        static void trtri(Vector<TConfig> &A);

        static void sygv(Vector<TConfig> &A, Vector<TConfig> &B,
                         Vector<TConfig> &eigenvalues, Vector<TConfig> &work);

        static void geqrf(Vector<TConfig> &A, Vector<TConfig> &tau, Vector<TConfig> &work);

        static void orgqr(Vector<TConfig> &A, Vector<TConfig> &tau, Vector<TConfig> &work);

        static void syevd(Vector<TConfig> &A, Vector<TConfig> &eigenvalues);

        // MAGMA only function.
        // dwork is a *device* workspace.
        static void stedx(Vector<TConfig> &diagonal,
                          Vector<TConfig> &subdiagonal,
                          Vector<TConfig> &eigenvectors,
                          int dim, Vector<TConfig_d> &dwork);
};

// Specialization for device.
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class Lapack< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    private:
        Lapack();
        ~Lapack();
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;

        static void check_magma_enabled();
        static void not_implemented();

        static void geqrf(Vector<TConfig> &A, Vector<TConfig_h> &tau, Vector<TConfig> &work);
        static void orgqr(Vector<TConfig> &A, Vector<TConfig_h> &tau, Vector<TConfig> &work);

        static void trtri(Vector<TConfig> &A);

        static void sygv(Vector<TConfig> &A, Vector<TConfig> &B,
                         Vector<TConfig_h> &eigenvalues, Vector<TConfig> &work);
        static void syevd(Vector<TConfig> &A, Vector<TConfig_h> &eigenvalues);
    private:
        // Unimplemented operations: private methods.
        static void geev(const Vector<TConfig> &A, Vector<TConfig> &eigenvalues);

};

}
