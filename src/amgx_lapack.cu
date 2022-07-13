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

#include <amgx_lapack.h>
#include <algorithm>

#ifdef AMGX_USE_MAGMA
#define ADD_ 1
#define HAVE_CUBLAS 1
#include <magma.h>
#endif

#include <amgx_cublas.h>

namespace amgx
{

#define lapackCheckError(status)                                \
    {                                                           \
        if (status < 0)                                         \
        {                                                       \
            std::stringstream ss;                               \
            ss << "Lapack error: argument number "              \
               << -status << " had an illegal value.";          \
            FatalError(ss.str(), AMGX_ERR_INTERNAL);           \
        }                                                       \
        else if (status > 0)                                    \
            FatalError("Lapack error: internal error.",         \
                       AMGX_ERR_INTERNAL);                     \
    }                                                           \

#define magmaCheckError(status)                                 \
    {                                                           \
        if (status < 0)                                         \
        {                                                       \
            std::stringstream ss;                               \
            ss << "Magma error: argument number "               \
               << -status << " had an illegal value.";          \
            FatalError(ss.str(), AMGX_ERR_INTERNAL);           \
        }                                                       \
        else if (status > 0)                                    \
            FatalError("Magma error: internal error.",          \
                       AMGX_ERR_INTERNAL);                     \
    }                                                           \

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::check_lapack_enabled()
{
#ifndef AMGX_USE_LAPACK
    FatalError("Error: LAPACK not enabled.", AMGX_ERR_CONFIGURATION);
#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::check_magma_enabled()
{
#ifndef AMGX_USE_MAGMA
    FatalError("Error: MAGMA not enabled.", AMGX_ERR_CONFIGURATION);
#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::check_magma_enabled()
{
#ifndef AMGX_USE_MAGMA
    FatalError("Error: MAGMA not enabled.", AMGX_ERR_CONFIGURATION);
#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::not_implemented()
{
    FatalError("Error: LAPACK operation not implemented on host.", AMGX_ERR_CONFIGURATION);
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::not_implemented()
{
    FatalError("Error: LAPACK operation not implemented on device.", AMGX_ERR_CONFIGURATION);
}

namespace
{

#ifdef AMGX_USE_LAPACK
struct _fcomplex { float re, im; };
typedef struct _fcomplex fcomplex;
struct _dcomplex { double re, im; };
typedef struct _dcomplex dcomplex;

extern "C"
int dgeev_(char *jobvl, char *jobvr, int *n, double *a,
           int *lda, double *wr, double *wi, double *vl,
           int *ldvl, double *vr, int *ldvr, double *work,
           int *lwork, int *info);

extern "C"
int sgeev_(char *jobvl, char *jobvr, int *n, float *a,
           int *lda, float *wr, float *wi, float *vl,
           int *ldvl, float *vr, int *ldvr, float *work,
           int *lwork, int *info);

extern "C"
int cgeev_(char *jobvl, char *jobvr, int *n, fcomplex *a,
           int *lda, fcomplex *wr, fcomplex *wi, fcomplex *vl,
           int *ldvl, fcomplex *vr, int *ldvr, fcomplex *work,
           int *lwork, int *info);

extern "C"
int zgeev_(char *jobvl, char *jobvr, int *n, dcomplex *a,
           int *lda, dcomplex *wr, dcomplex *wi, dcomplex *vl,
           int *ldvl, dcomplex *vr, int *ldvr, dcomplex *work,
           int *lwork, int *info);

int lapack_geev_dispatch(char *jobvl, char *jobvr, int *n, double *a,
                         int *lda, double *wr, double *wi, double *vl,
                         int *ldvl, double *vr, int *ldvr, double *work,
                         int *lwork, int *info)
{
    return dgeev_(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
}

int lapack_geev_dispatch(char *jobvl, char *jobvr, int *n, float *a,
                         int *lda, float *wr, float *wi, float *vl,
                         int *ldvl, float *vr, int *ldvr, float *work,
                         int *lwork, int *info)
{
    return sgeev_(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
}

int lapack_geev_dispatch(char *jobvl, char *jobvr, int *n, cuComplex *a,
                         int *lda, cuComplex *wr, cuComplex *wi, cuComplex *vl,
                         int *ldvl, cuComplex *vr, int *ldvr, cuComplex *work,
                         int *lwork, int *info)
{
    return cgeev_(jobvl, jobvr, n, 
                  reinterpret_cast<fcomplex *>(a),
                  lda, 
                  reinterpret_cast<fcomplex *>(wr), 
                  reinterpret_cast<fcomplex *>(wi), 
                  reinterpret_cast<fcomplex *>(vl),
                  ldvl, 
                  reinterpret_cast<fcomplex *>(vr), 
                  ldvr, 
                  reinterpret_cast<fcomplex *>(work), 
                  lwork,
                  info);
}

int lapack_geev_dispatch(char *jobvl, char *jobvr, int *n, cuDoubleComplex *a,
                         int *lda, cuDoubleComplex *wr, cuDoubleComplex *wi, cuDoubleComplex *vl,
                         int *ldvl, cuDoubleComplex *vr, int *ldvr, cuDoubleComplex *work,
                         int *lwork, int *info)
{
    return zgeev_(jobvl, jobvr, n, 
                  reinterpret_cast<dcomplex *>(a),
                  lda, 
                  reinterpret_cast<dcomplex *>(wr), 
                  reinterpret_cast<dcomplex *>(wi), 
                  reinterpret_cast<dcomplex *>(vl),
                  ldvl, 
                  reinterpret_cast<dcomplex *>(vr), 
                  ldvr, 
                  reinterpret_cast<dcomplex *>(work), 
                  lwork,
                  info);
}

template <typename T>
void lapack_geev(T *A, T *eigenvalues, int dim, int lda)
{
    char job = 'N';
    T *WI = new T[dim];
    int ldv = 1;
    T *vl = 0;
    int work_size = 6 * dim;
    T *work = new T[work_size];
    int info;
    lapack_geev_dispatch(&job, &job, &dim, A, &lda, eigenvalues, WI, vl, &ldv,
                         vl, &ldv, work, &work_size, &info);
    lapackCheckError(info);
    delete [] WI;
    delete [] work;
}

template <typename T>
void lapack_geev(T *A, T *eigenvalues, T *eigenvectors, int dim, int lda, int ldvr)
{
    char jobvl = 'N';
    char jobvr = 'V';
    T *WI = new T[dim * dim];
    int work_size = 6 * dim;
    T *vl = 0;
    int ldvl = 1;
    T *work = new T[work_size];
    int info;
    lapack_geev_dispatch(&jobvl, &jobvr, &dim, A, &lda, eigenvalues, WI, vl, &ldvl,
                         eigenvectors, &ldvr, work, &work_size, &info);
    lapackCheckError(info);
    delete [] WI;
    delete [] work;
}

#endif

} // end anonymous namespace

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::geev(const Vector<TConfig> &A, Vector<TConfig> &eigenvalues)
{
    check_lapack_enabled();
    typedef typename Vector<TConfig>::value_type value_type;
    // It is possible the matrix has an extra row (e.g. Arnoldi).
    int dim = std::min(A.get_num_rows(), A.get_num_cols());
    int lda = A.get_lda();
    value_type *A_ptr = const_cast<value_type *>(A.raw());
#ifdef AMGX_USE_LAPACK
    lapack_geev(A_ptr, eigenvalues.raw(), dim, lda);
#else
    FatalError("Lapack is not supported in this build", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::geev(const Vector<TConfig> &A,
        Vector<TConfig> &eigenvalues,
        Vector<TConfig> &eigenvector)
{
    check_lapack_enabled();
    typedef typename Vector<TConfig>::value_type value_type;
    // It is possible the matrix has an extra row (e.g. Arnoldi).
    int dim = std::min(A.get_num_rows(), A.get_num_cols());
    int lda = A.get_lda();
    value_type *A_ptr = const_cast<value_type *>(A.raw());
#ifdef AMGX_USE_LAPACK
    lapack_geev(A_ptr, eigenvalues.raw(), eigenvector.raw(), dim, lda, eigenvector.get_lda());
#else
    FatalError("Lapack is not supported in this build", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}


template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::geev(const Vector<TConfig> &A, Vector<TConfig> &eigenvalues)
{
    not_implemented();
}

namespace
{
#ifdef AMGX_USE_LAPACK
extern "C"
int dtrtri_(char *uplo, char *diag, int *n, double *
            a, int *lda, int *info);

extern "C"
int strtri_(char *uplo, char *diag, int *n, float *
            a, int *lda, int *info);

extern "C"
int ctrtri_(char *uplo, char *diag, int *n, fcomplex *
            a, int *lda, int *info);

extern "C"
int ztrtri_(char *uplo, char *diag, int *n, dcomplex *
            a, int *lda, int *info);

int lapack_trtri_dispatch(char *uplo, char *diag, int *n, float *a,
                          int *lda, int *info)
{
    return strtri_(uplo, diag, n, a, lda, info);
}

int lapack_trtri_dispatch(char *uplo, char *diag, int *n, double *a,
                          int *lda, int *info)
{
    return dtrtri_(uplo, diag, n, a, lda, info);
}

int lapack_trtri_dispatch(char *uplo, char *diag, int *n, fcomplex *a,
                          int *lda, int *info)
{
    return ctrtri_(uplo, diag, n, a, lda, info);
}

int lapack_trtri_dispatch(char *uplo, char *diag, int *n, dcomplex *a,
                          int *lda, int *info)
{
    return ztrtri_(uplo, diag, n, a, lda, info);
}

template <typename T>
void lapack_trtri(T *A, int dim, int lda)
{
    char uplo = 'U';
    char diag = 'N';
    int info;
    lapack_trtri_dispatch(&uplo, &diag, &dim, A, &lda, &info);
    lapackCheckError(info);
}
#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::trtri(Vector<TConfig> &A)
{
    check_lapack_enabled();
    typedef typename Vector<TConfig>::value_type value_type;
    int dim = std::min(A.get_num_rows(), A.get_num_cols());
    int lda = A.get_lda();
#ifdef AMGX_USE_LAPACK
    lapack_trtri(A.raw(), dim, lda);
#else
    FatalError("Lapack is not supported in this build", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

namespace
{
#ifdef AMGX_USE_MAGMA
int magma_trtri_dispatch(magma_uplo_t uplo, magma_diag_t diag, int n, float *a,
                         int lda, int *info)
{
    return magma_strtri_gpu(uplo, diag, n, a, lda, info);
}

int magma_trtri_dispatch(magma_uplo_t uplo, magma_diag_t diag, int n, double *a,
                         int lda, int *info)
{
    return magma_dtrtri_gpu(uplo, diag, n, a, lda, info);
}

int magma_trtri_dispatch(magma_uplo_t uplo, magma_diag_t diag, int n, cuComplex *a,
                         int lda, int *info)
{
    return magma_ctrtri_gpu(uplo, diag, n, a, lda, info);
}

int magma_trtri_dispatch(magma_uplo_t uplo, magma_diag_t diag, int n, cuDoubleComplex *a,
                         int lda, int *info)
{
    return magma_ztrtri_gpu(uplo, diag, n, a, lda, info);
}

template <typename T>
void magma_trtri(T *A, int dim, int lda)
{
    magma_uplo_t uplo = MagmaUpper;
    magma_diag_t diag = MagmaNonUnit;
    int info;
    magma_trtri_dispatch(uplo, diag, dim, A, lda, &info);
    magmaCheckError(info);
}

#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::trtri(Vector<TConfig> &A)
{
    check_magma_enabled();
    typedef typename Vector<TConfig>::value_type value_type;
    int dim = std::min(A.get_num_rows(), A.get_num_cols());
    int lda = A.get_lda();
#ifdef AMGX_USE_MAGMA
    magma_trtri(A.raw(), dim, lda);;
#else
    FatalError("Lapack is not supported in this build", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

namespace
{
#ifdef AMGX_USE_LAPACK
extern "C"
int dsygv_(int *itype, char *jobz, char *uplo, int *n,
           double *a, int *lda, double *b, int *ldb,
           double *w, double *work, int *lwork, int *info);

extern "C"
int ssygv_(int *itype, char *jobz, char *uplo, int *n,
           float *a, int *lda, float *b, int *ldb,
           float *w, float *work, int *lwork, int *info);

extern "C"
int chegv_(int *itype, char *jobz, char *uplo, int *n,
           fcomplex *a, int *lda, fcomplex *b, int *ldb,
           fcomplex *w, fcomplex *work, int *lwork, int *info);

extern "C"
int zhegv_(int *itype, char *jobz, char *uplo, int *n,
           dcomplex *a, int *lda, dcomplex *b, int *ldb,
           dcomplex *w, dcomplex *work, int *lwork, int *info);

int lapack_sygv_dispatch(int *itype, char *jobz, char *uplo, int *n,
                         double *a, int *lda, double *b, int *ldb,
                         double *w, double *work, int *lwork, int *info)
{
    return dsygv_(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, info);
}

int lapack_sygv_dispatch(int *itype, char *jobz, char *uplo, int *n,
                         float *a, int *lda, float *b, int *ldb,
                         float *w, float *work, int *lwork, int *info)
{
    return ssygv_(itype, jobz, uplo, n, a, lda, b, ldb, w, work, lwork, info);
}

int lapack_sygv_dispatch(int *itype, char *jobz, char *uplo, int *n,
                         cuComplex *a, int *lda, cuComplex *b, int *ldb,
                         cuComplex *w, cuComplex *work, int *lwork, int *info)
{
    return chegv_(itype, jobz, uplo, n,
                  reinterpret_cast<fcomplex *>(a), lda, reinterpret_cast<fcomplex *>(b), ldb,
                  reinterpret_cast<fcomplex *>(w), reinterpret_cast<fcomplex *>(work), lwork, info);
}

int lapack_sygv_dispatch(int *itype, char *jobz, char *uplo, int *n,
                         cuComplex *a, int *lda, cuDoubleComplex *b, int *ldb,
                         cuDoubleComplex *w, cuDoubleComplex *work, int *lwork, int *info)
{
    return zhegv_(itype, jobz, uplo, n,
                  reinterpret_cast<dcomplex *>(a), lda, reinterpret_cast<dcomplex *>(b), ldb,
                  reinterpret_cast<dcomplex *>(w), reinterpret_cast<dcomplex *>(work), lwork, info);
}

template <typename T>
void lapack_sygv(T *gramA, T *gramB, T *eigenvector, int dim, int lda, T *work)
{
    int itype = 1;
    char jobz = 'V';
    char uplo = 'U';
    int ldb = lda;
    int lwork = 1024;
    int info = 0;
    lapack_sygv_dispatch(&itype, &jobz, &uplo, &dim, gramA, &lda, gramB, &ldb, eigenvector, work, &lwork, &info);
    lapackCheckError(info);
}
#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::sygv(Vector<TConfig> &A, Vector<TConfig> &B,
        Vector<TConfig> &eigenvalues, Vector<TConfig> &work)
{
    check_lapack_enabled();
    typedef typename Vector<TConfig>::value_type value_type;
    int dim = std::min(A.get_num_rows(), A.get_num_cols());
    int lda = A.get_lda();
#ifdef AMGX_USE_LAPACK
    lapack_sygv(A.raw(), B.raw(), eigenvalues.raw(), dim, lda, work.raw());
#else
    FatalError("Lapack is not supported in this build", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

namespace
{
#ifdef AMGX_USE_MAGMA
void magma_trsm_dispatch(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                         magma_diag_t diag, magma_int_t m, magma_int_t n,
                         float alpha, float const *dA, magma_int_t lda,
                         float *dB, magma_int_t ldb)
{
    return magma_strsm(side, uplo, trans, diag, m, n, alpha, dA, lda, dB, ldb);
}

void magma_trsm_dispatch(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                         magma_diag_t diag, magma_int_t m, magma_int_t n,
                         double alpha, double const *dA, magma_int_t lda,
                         double *dB, magma_int_t ldb)
{
    return magma_dtrsm(side, uplo, trans, diag, m, n, alpha, dA, lda, dB, ldb);
}

void magma_trmm_dispatch(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                         magma_diag_t diag, magma_int_t m, magma_int_t n,
                         float alpha, float const *dA, magma_int_t lda,
                         float *dB, magma_int_t ldb)
{
    return magma_strmm(side, uplo, trans, diag, m, n, alpha, dA, lda, dB, ldb);
}

void magma_trmm_dispatch(magma_side_t side, magma_uplo_t uplo, magma_trans_t trans,
                         magma_diag_t diag, magma_int_t m, magma_int_t n,
                         double alpha, double const *dA, magma_int_t lda,
                         double *dB, magma_int_t ldb)
{
    return magma_dtrmm(side, uplo, trans, diag, m, n, alpha, dA, lda, dB, ldb);
}

int magma_potrf_gpu_dispatch(magma_uplo_t uplo, int n, float *A, int lda, int *info)
{
    return magma_spotrf_gpu(uplo, n, A, lda, info);
}

int magma_potrf_gpu_dispatch(magma_uplo_t uplo, int n, double *A, int lda, int *info)
{
    return magma_dpotrf_gpu(uplo, n, A, lda, info);
}

int magma_sygst_gpu_dispatch(int itype, magma_uplo_t uplo, magma_int_t n, float *da,
                             int ldda, float *B, int lddb, int *info)
{
    return magma_ssygst_gpu(itype, uplo, n, da, ldda, B, lddb, info);
}

int magma_sygst_gpu_dispatch(int itype, magma_uplo_t uplo, magma_int_t n, double *da,
                             int ldda, double *B, int lddb, int *info)
{
    return magma_dsygst_gpu(itype, uplo, n, da, ldda, B, lddb, info);
}

int magma_syevd_gpu_dispatch(magma_vec_t jobz, magma_uplo_t uplo, int n, double *da, int ldda,
                             double *w, double *wa, int ldwa, double *work,
                             int lwork, int *iwork, int liwork, int *info)
{
    return magma_dsyevd_gpu(jobz, uplo, n, da, ldda, w, wa, ldwa, work, lwork, iwork, liwork, info);
}

int magma_syevd_gpu_dispatch(magma_vec_t jobz, magma_uplo_t uplo, int n, float *da, int ldda,
                             float *w, float *wa, int ldwa, float *work,
                             int lwork, int *iwork, int liwork, int *info)
{
    return magma_ssyevd_gpu(jobz, uplo, n, da, ldda, w, wa, ldwa, work, lwork, iwork, liwork, info);
}

// This is a simple modification of the magma_?sygvd() source code
// from magma where the matrices are already on the device.
template <typename T>
magma_int_t magma_sygvd_gpu_impl(magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
                                 T *da, magma_int_t lda, T *db, magma_int_t ldb, T *w,
                                 T *work, magma_int_t lwork, magma_int_t *iwork, magma_int_t liwork,
                                 T *wa, magma_int_t *info)
{
    magma_uplo_t uplo_[2] = {uplo, MagmaLower}; // {uplo, 0}
    magma_vec_t jobz_[2] = {jobz, MagmaVec};//{jobz, 0};
    T d_one = MAGMA_D_ONE;
    magma_int_t ldda = n;
    magma_int_t lddb = n;
    static magma_int_t lower;
    static char trans[1];
    static magma_int_t wantz, lquery;
    static magma_int_t lopt, lwmin, liopt, liwmin;
    static cudaStream_t stream;
    magma_queue_create( &stream );
    wantz = jobz_[0] == MagmaVec;
    lower = uplo_[0] == MagmaLower;
    lquery = lwork == -1 || liwork == -1;
    *info = 0;

    if (itype < 1 || itype > 3)
    {
        *info = -1;
    }
    else if (! (wantz || jobz_[0] == MagmaNoVec))
    {
        *info = -2;
    }
    else if (! (lower || uplo_[0] == MagmaUpper))
    {
        *info = -3;
    }
    else if (n < 0)
    {
        *info = -4;
    }
    else if (lda < max(1, n))
    {
        *info = -6;
    }
    else if (ldb < max(1, n))
    {
        *info = -8;
    }

    magma_int_t nb = magma_get_dsytrd_nb(n);

    if (n < 1)
    {
        liwmin = 1;
        lwmin = 1;
    }
    else if (wantz)
    {
        lwmin = 1 + 6 * n * nb + 2 * n * n;
        liwmin = 5 * n + 3;
    }
    else
    {
        lwmin = 2 * n * nb + 1;
        liwmin = 1;
    }

    lopt = lwmin;
    liopt = liwmin;
    work[ 0] = lopt;
    iwork[0] = liopt;

    if (lwork < lwmin && ! lquery)
    {
        *info = -11;
    }
    else if (liwork < liwmin && ! lquery)
    {
        *info = -13;
    }

    if (*info != 0)
    {
        magma_xerbla( __func__, -(*info) );
        return MAGMA_ERR_ILLEGAL_VALUE;
    }
    else if (lquery)
    {
        return MAGMA_SUCCESS;
    }

    /* Quick return if possible */
    if (n == 0)
    {
        return 0;
    }

    magma_potrf_gpu_dispatch(uplo_[0], n, db, lddb, info);

    if (*info != 0)
    {
        *info = n + *info;
        return 0;
    }

    /* Transform problem to standard eigenvalue problem and solve. */
    magma_sygst_gpu_dispatch(itype, uplo_[0], n, da, ldda, db, lddb, info);
    magma_syevd_gpu_dispatch(jobz_[0], uplo_[0], n, da, ldda, w, wa, lda,
                             work, lwork, iwork, liwork, info);
    lopt = max( lopt, (magma_int_t) work[0]);
    liopt = max(liopt, iwork[0]);

    if (wantz && *info == 0)
    {
        /* Backtransform eigenvectors to the original problem. */
        if (itype == 1 || itype == 2)
        {
            /* For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
               backtransform eigenvectors: x = inv(L)'*y or inv(U)*y */
            if (lower)
            {
                *(unsigned char *)trans = MagmaTrans;
            }
            else
            {
                *(unsigned char *)trans = MagmaNoTrans;
            }

            magma_trsm_dispatch(MagmaLeft, uplo_[0], *trans, MagmaNonUnit,
                                n, n, d_one, db, lddb, da, ldda);
        }
        else if (itype == 3)
        {
            /* For B*A*x=(lambda)*x;
               backtransform eigenvectors: x = L*y or U'*y */
            if (lower)
            {
                *(unsigned char *)trans = MagmaNoTrans;
            }
            else
            {
                *(unsigned char *)trans = MagmaTrans;
            }

            magma_trmm_dispatch(MagmaLeft, uplo_[0], *trans, MagmaNonUnit,
                                n, n, d_one, db, lddb, da, ldda);
        }
    }

    magma_queue_sync( stream );
    magma_queue_destroy( stream );
    work[0] = (T) lopt;
    iwork[0] = liopt;
    return MAGMA_SUCCESS;
}

cublasStatus_t cublas_trsm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const float *alpha,
                           const float *A, int lda,
                           float *B, int ldb)
{
    return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

cublasStatus_t cublas_trsm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const double *alpha,
                           const double *A, int lda,
                           double *B, int ldb)
{
    return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}


template <typename T>
void magma_sygvd_gpu(T *A, T *B, T *eigenvalues, int dim, int lda)
{
    int itype = 1;
    magma_vec_t jobz = MagmaVec;
    magma_uplo_t uplo = MagmaUpper;
    int N = dim;
    int ldb = lda;
    int nb = 32;
    int lwork = 1 + 6 * N * nb + 2 * N * N;
    static std::vector<T> s_work;
    s_work.resize(lwork);
    T *work = &s_work[0];
    int liwork = 3 + 5 * N;
    static std::vector<int> s_iwork;
    s_iwork.resize(liwork);
    int *iwork = &s_iwork[0];
    static std::vector<T> s_wa;
    s_wa.resize(lda * N);
    T *wa = &s_wa[0];
    int ldwa = N;
    int info;
    /*
            magma_sygvd_gpu_impl(itype, jobz, uplo, N, A, lda, B, ldb, eigenvalues, work, lwork, iwork, liwork, wa, &info);
    */
    magma_potrf_gpu_dispatch(uplo, N, B, lda, &info);
    magmaCheckError(info);
    magma_sygst_gpu_dispatch(itype, uplo, N, A, lda, B, ldb, &info);
    magmaCheckError(info);
    magma_syevd_gpu_dispatch(jobz, uplo, N, A, lda, eigenvalues, wa, ldwa, work, lwork, iwork, liwork, &info);
    magmaCheckError(info);
    T one = 1;
    cublasHandle_t handle = Cublas::get_handle();
    cublas_trsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, N, &one, B, ldb, A, lda);
}
#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::sygv(Vector<TConfig> &A, Vector<TConfig> &B,
        Vector<TConfig_h> &eigenvalues, Vector<TConfig> &work)
{
    typedef typename Vector<TConfig>::value_type value_type;
    int dim = std::min(A.get_num_rows(), A.get_num_cols());
    int lda = A.get_lda();
#ifdef AMGX_USE_MAGMA
    magma_sygvd_gpu(A.raw(), B.raw(), eigenvalues.raw(), dim, lda);
#endif
}

namespace
{
#ifdef AMGX_USE_MAGMA
template <typename T>
void magma_syevd_gpu(T *A, T *eigenvalues, int dim, int lda)
{
    magma_vec_t jobz = MagmaVec;
    magma_uplo_t uplo = MagmaUpper;
    int N = dim;
    int nb = 32;
    int lwork = 1 + 6 * N * nb + 2 * N * N;
    static std::vector<T> s_work;
    s_work.resize(lwork);
    T *work = &s_work[0];
    int liwork = 3 + 5 * N;
    static std::vector<int> s_iwork;
    s_iwork.resize(liwork);
    int *iwork = &s_iwork[0];
    static std::vector<T> s_wa;
    s_wa.resize(lda * N);
    T *wa = &s_wa[0];
    int ldwa = N;
    int info;
    magma_syevd_gpu_dispatch(jobz, uplo, N, A, lda, eigenvalues, wa, ldwa, work, lwork, iwork, liwork, &info);
    magmaCheckError(info);
}
#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::syevd(Vector<TConfig> &A,
        Vector<TConfig_h> &eigenvalues)
{
    check_magma_enabled();
    typedef typename Vector<TConfig>::value_type value_type;
    int dim = std::min(A.get_num_rows(), A.get_num_cols());
    int lda = A.get_lda();
#ifdef AMGX_USE_MAGMA
    magma_syevd_gpu(A.raw(), eigenvalues.raw(), dim, lda);
#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::syevd(Vector<TConfig> &A,
        Vector<TConfig> &eigenvalues)
{
    not_implemented();
}

namespace
{
#ifdef AMGX_USE_MAGMA
int magma_stedx_dispatch(magma_range_t range, int n,
                         double vl, double vu,
                         int il, int iu,
                         double *d, double *e, double *z, int ldz,
                         double *work, int lwork, int *iwork, int liwork,
                         double *dwork, int *info)
{
    return magma_dstedx(range, n, vl, vu, il, iu, d, e, z, ldz, work, lwork, iwork, liwork, dwork, info);
}

int magma_stedx_dispatch(magma_range_t range, int n,
                         float vl, float vu,
                         int il, int iu,
                         float *d, float *e, float *z, int ldz,
                         float *work, int lwork, int *iwork, int liwork,
                         float *dwork, int *info)
{
    return magma_sstedx(range, n, vl, vu, il, iu, d, e, z, ldz, work, lwork, iwork, liwork, dwork, info);
}

template <typename T>
void magma_stedx(T *diagonal, T *subdiagonal, T *eigenvectors,
                 int lower, int upper, int dim, int ldz, T *dwork, int dwork_size)
{
    magma_range_t range = MagmaRangeI;
    int N = dim;
    T vl = 0;
    T vu = 0;
    int il = lower;
    int iu = upper;
    int lwork = 1 + 4 * N + 2 * N * N;
    static std::vector<T> s_work;
    s_work.resize(lwork);
    int liwork = 3 + 6 * N;
    static std::vector<int> s_iwork;
    s_iwork.resize(liwork);
    int info;
    magma_stedx_dispatch(range, N, vl, vu, il, iu, diagonal, subdiagonal, eigenvectors, ldz,
                         &s_work[0], lwork, &s_iwork[0], liwork, dwork, &info);
}
#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::stedx(Vector<TConfig> &diagonal,
        Vector<TConfig> &subdiagonal,
        Vector<TConfig> &eigenvectors,
        int dim,
        Vector<TConfig_d> &dwork)
{
    check_magma_enabled();
#ifdef AMGX_USE_MAGMA
    magma_stedx(diagonal.raw(), subdiagonal.raw(), eigenvectors.raw(),
                dim, dim, dim, eigenvectors.get_lda(),
                dwork.raw(), dwork.size());
#endif
}


namespace
{

template <typename T>
void larf(int m, int n, T *v,
          int incv, T *tau, T *c, int ldc,
          T *work)
{
    /* Table of constant values */
    static T c_b4 = 1.;
    static T c_b5 = 0.;
    static int c1 = 1;
    /*        Form  H * C */
    /*        w := C' * v */
    Cublas::gemv(true, m, n, &c_b4, c, ldc,
                 v, incv, &c_b5, work, c1);
    /*        C := C - v * w' */
    Cublas::ger(m, n, tau, v, incv, work, c1, c, ldc);
}

template <typename T>
__global__
void set1(T *a)
{
    *a = 1.;
}

template <typename T>
__global__
void add_tau(T *a, T tau)
{
    *a = 1 + tau;
}

template <typename T>
void gpu_orgqr(int m, int n, int k,
               T *a, int lda, T *tau, T *work, int lwork)
{
    int i1, i2;

    for (int i = k - 1; i >= 0; --i)
    {
        /*        Apply H(i) to A(i:m,i:n) from the left */
        if (i < n - 1)
        {
            set1 <<< 1, 1>>>(&a[i + i * lda]);
            i1 = m - i;
            i2 = n - i - 1;
            larf(i1, i2, &a[i + i * lda], 1, &tau[i],
                 &a[i + (i + 1) * lda], lda, work);
        }

        if (i < m - 1)
        {
            i1 = m - i - 1;
            Cublas::scal(i1, &tau[i], &a[i + 1 + i * lda], 1);
        }

        add_tau <<< 1, 1>>>(&a[i + i * lda], tau[i]);
        /*        Set A(1:i-1,i) to zero */
        cudaMemset(&a[i * lda], 0, sizeof(T) * i);
    }

    cudaCheckError();
}

template <typename T>
__device__ __host__
T lapy2_(T *a, T *b)
{
    T va = *a;
    T vb = *b;
    return sqrt(va * va + vb * vb);
}

template <typename T>
__device__ __host__
T d_sign(T a, T b)
{
    T x;
    x = (a >= 0 ? a : - a);
    return (b >= 0 ? x : -x);
}

template <typename T>
void compute_tau_host(T *alpha, T *norm,
                      T *tau, T *d1)
{
    *d1 = lapy2_(alpha, norm);
    T beta = -d_sign(*d1, *alpha);
    // LAPACK: skipped part about scaling.
    // Negated compared to LAPACK code, avoid negating value on device later.
    *tau = -(beta - *alpha) / beta;
    *d1 = 1. / (*alpha - beta);
    *alpha = beta;
}

template <typename T>
void larfg(int n, T *alpha, T *x,
           int incx, T *tau)
{
    if (n <= 1)
    {
        *tau = 0.;
        return;
    }

    int i1 = n - 1;
    T xnorm;
    Cublas::nrm2(i1, x, incx, &xnorm);
    T h_alpha;
    cudaMemcpy(&h_alpha, alpha, sizeof(T), cudaMemcpyDeviceToHost);
    T d1;
    compute_tau_host(&h_alpha, &xnorm, tau, &d1);
    Cublas::scal(i1, d1, x, incx);
    // Update the diagonal value on the device.
    cudaMemcpy(alpha, &h_alpha, sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void gpu_geqrf(int m, int n, T *a, int lda,
               T *tau, T *work)
{
    int k = std::min(m, n);
    T *aii;
    cudaMallocAsync(&aii, sizeof(T), 0);

    for (int i = 0; i < k; ++i)
    {
        /*        Generate elementary reflector H(i) to annihilate A(i+1:m,i) */
        int i2 = m - i;
        /* Computing MIN */
        int i3 = i + 1;
        larfg(i2, &a[i + i * lda],
              &a[std::min(i3, m - 1) + i * lda],
              1, &tau[i]);

        if (i < n - 1)
        {
            /*           Apply H(i) to A(i:m,i+1:n) from the left */
            cudaMemcpy(aii, &a[i + i * lda], sizeof(T), cudaMemcpyDeviceToDevice);
            set1 <<< 1, 1>>>(&a[i + i * lda]);
            cudaCheckError();
            i2 = m - i;
            i3 = n - i - 1;
            larf(i2, i3, &a[i + i * lda], 1,
                 &tau[i], &a[i + (i + 1) * lda], lda, work);
            cudaMemcpy(&a[i + i * lda], aii, sizeof(T), cudaMemcpyDeviceToDevice);
        }
    }

    cudaFreeAsync(aii, 0);
}
} // end anonymous namespace

namespace
{
#ifdef AMGX_USE_MAGMA
int magma_geqrf_dispatch(int m, int n, float *A, int lda,
                         float *tau, float *work, int *info)
{
    return magma_sgeqrf_gpu(m, n, A, lda, tau, work, info);
}

int magma_geqrf_dispatch(int m, int n, double *A, int lda,
                         double *tau, double *work, int *info)
{
    return magma_dgeqrf_gpu(m, n, A, lda, tau, work, info);
}

template <typename T>
void magma_geqrf(int m, int n, T *A, int lda,
                 T *tau, T *work)
{
    int info;
    magma_geqrf_dispatch(m, n, A, lda, tau, work, &info);
    magmaCheckError(info);
}

int magma_orgqr_dispatch(int m, int n, int k, float *A, int lda,
                         float *tau, float *work, int lwork, int *info)
{
    return magma_sorgqr_gpu(m, n, k, A, lda, tau, work, lwork, info);
}

int magma_orgqr_dispatch(int m, int n, int k, double *A, int lda,
                         double *tau, double *work, int lwork, int *info)
{
    return magma_dorgqr_gpu(m, n, k, A, lda, tau, work, lwork, info);
}

template <typename T>
void magma_orgqr(int m, int n, int k, T *A, int lda,
                 T *tau, T *work, int lwork)
{
    int info;
    magma_orgqr_dispatch(m, n, k, A, lda, tau, work, lwork, &info);
    magmaCheckError(info);
}
#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::geqrf(Vector<TConfig> &A,
        Vector<TConfig> &tau,
        Vector<TConfig> &work)
{
    not_implemented();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::geqrf(Vector<TConfig> &A,
        Vector<TConfig_h> &tau,
        Vector<TConfig> &work)
{
    int rows = A.get_num_rows();
    int cols = A.get_num_cols();
    int lda = A.get_lda();
#ifdef AMGX_USE_MAGMA
    magma_geqrf(rows, cols, A.raw(), lda, tau.raw(), work.raw());
#else
    gpu_geqrf(rows, cols, A.raw(), lda, tau.raw(), work.raw());
#endif
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::orgqr(Vector<TConfig> &A,
        Vector<TConfig> &tau,
        Vector<TConfig> &work)
{
    not_implemented();
}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void Lapack< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::orgqr(Vector<TConfig> &A,
        Vector<TConfig_h> &tau,
        Vector<TConfig> &work)
{
    int rows = A.get_num_rows();
    int cols = A.get_num_cols();
    int lda = A.get_lda();
#ifdef AMGX_USE_MAGMA
    magma_orgqr(rows, cols, cols, A.raw(), lda, tau.raw(), work.raw(), 1);
#else
    gpu_orgqr(rows, cols, cols, A.raw(), lda, tau.raw(), work.raw(), 1);
#endif
}


#define AMGX_CASE_LINE(CASE) \
    template class Lapack<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

}
