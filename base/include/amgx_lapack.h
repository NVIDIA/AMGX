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
