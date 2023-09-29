/* Copyright (c) 2011-2017, NVIDIA CORPORATION. All rights reserved.
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

#include <string>
#include <cublas_v2.h>
#include <solvers/solver.h>

#include <amgx_cusolverDn.h>

namespace amgx
{
namespace dense_lu_solver
{

template< typename T_Config >
class DenseLUSolver
{};

template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class DenseLUSolver<TemplateConfig<AMGX_host, V, M, I> >
    : public Solver<TemplateConfig<AMGX_host, V, M, I> >
{

    public:

        typedef TemplateConfig<AMGX_host, V, M, I> Config_h;
        typedef Matrix<Config_h> Matrix_h;
        typedef Vector<Config_h> Vector_h;
        typedef typename MatPrecisionMap<M>::Type Matrix_data;
        typedef typename VecPrecisionMap<V>::Type Vector_data;

        DenseLUSolver(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng)
            : Solver<Config_h>(cfg, cfg_scope, tmng) {}
        virtual bool isColoringNeeded() const { return false; }
        virtual bool getReorderColsByColorDesired() const { return false; }
        virtual bool getInsertDiagonalDesired() const { return false; }
        virtual void solver_setup(bool reuse_matrix_structure)
        {
            FatalError("No host implementation of the dense LU solver", AMGX_ERR_NOT_IMPLEMENTED);
        }
        virtual void solve_init(Vector_h &b, Vector_h &x, bool xIsZero)
        {
            FatalError("No host implementation of the dense LU solver", AMGX_ERR_NOT_IMPLEMENTED);
        }
        virtual AMGX_STATUS solve_iteration(Vector_h &b, Vector_h &x, bool xIsZero)
        {
            FatalError("No host implementation of the dense LU solver", AMGX_ERR_NOT_IMPLEMENTED);
        }
        virtual void solve_finalize(Vector_h &b, Vector_h &x)
        {
            FatalError("No host implementation of the dense LU solver", AMGX_ERR_NOT_IMPLEMENTED);
        }
};


template< AMGX_VecPrecision V, AMGX_MatPrecision M, AMGX_IndPrecision I >
class DenseLUSolver<TemplateConfig<AMGX_device, V, M, I> >
    : public Solver<TemplateConfig<AMGX_device, V, M, I> >
{

    public:

        typedef Solver<TemplateConfig<AMGX_device, V, M, I> > Base;
        typedef TemplateConfig<AMGX_device, V, M, I> Config_d;
        typedef TemplateConfig<AMGX_host, V, M, I> Config_h;
        typedef Matrix<Config_d> Matrix_d;
        typedef Matrix<Config_h> Matrix_h;
        typedef Vector<Config_d> Vector_d;
        typedef Vector<Config_d> Vector_h;
        typedef typename Matrix_d::IVector IVector_d;
        typedef typename Matrix_h::IVector IVector_h;
        typedef typename Matrix_d::MVector MVector_d;
        typedef typename Matrix_h::MVector MVector_h;
        typedef typename MatPrecisionMap<M>::Type Matrix_data;
        typedef typename VecPrecisionMap<V>::Type Vector_data;

        DenseLUSolver(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng);
        virtual ~DenseLUSolver();
        virtual bool isColoringNeeded() const { return false; }
        virtual bool getReorderColsByColorDesired() const { return false; }
        virtual bool getInsertDiagonalDesired() const { return false; }
        virtual void solver_setup(bool reuse_matrix_structure);
        virtual void solve_init(Vector_d &b, Vector_d &x, bool xIsZero);
        virtual AMGX_STATUS solve_iteration(Vector_d &b, Vector_d &x, bool xIsZero);
        virtual void solve_finalize(Vector_d &b, Vector_d &x);

        inline int get_num_rows() const { return m_num_rows; }
        inline int get_lda() const { return m_lda; }
        inline const Matrix_data *get_dense_A() const { return m_dense_A; }

    private:

        cusolverDnHandle_t m_cuds_handle;
        cublasHandle_t m_cublas_handle;
        int m_num_rows, m_num_cols, m_lda;
        int m_nnz_global;
        Matrix_data *m_dense_A;   // store sparse as dense
        int *m_ipiv;              // The pivot sequence from getrf()
        int *m_cuds_info;         // host pointer for debug info from getrf()
        Matrix_data *m_trf_wspace; // workspace for trf/trs
        bool m_enable_exact_solve = false;

        // Cached in the case of an exact coarse solve
        IVector_h nz_all;
        IVector_h nz_displs;
        IVector_h row_all;
        IVector_h row_displs;
        IVector_d Acols_global;
        IVector_d Arows_global;

        void csr_to_dense(); // Pack a CSR matrix to a dense matrix
        void cudense_getrf(); // LU decomposition
        void cudense_getrs(Vector_d &x); // solve using LU from getrf()
        template< class DataType, class IndexType > void
        allocMem(DataType *&ptr, IndexType numEntry, bool initToZero = false);
        // Modify the rhs to include contribution from halo nodes
        void distributed_rhs_mod(const Vector_d &b,
                                 const Vector_d &x, Vector_d &new_rhs);
};

template< class T_Config >
class DenseLUSolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create(AMG_Config &cfg,
                                 const std::string &cfg_scope,
                                 ThreadManager *tmng)
        {
            return new DenseLUSolver<T_Config>(cfg, cfg_scope, tmng);
        }
};

} // namespace dense_lu_solver
} // namespace amgx

