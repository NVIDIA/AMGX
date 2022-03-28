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

namespace amgx
{
template <class T_Config> class DistributedArranger;
}

#include <cstdio>

#include <amgx_cusparse.h>
#include <thrust/sequence.h>
#include <vector.h>
#include <error.h>

namespace amgx
{

template <class T_Config> class DistributedManager;
template <class T_Config> class DistributedManagerBase;
template <class T_Config> class Matrix;

typedef typename IndPrecisionMap<AMGX_indInt>::Type INDEX_TYPE;

template <class T_Config> class DistributedArrangerBase
{
    public:
        typedef T_Config TConfig;
        typedef typename Matrix<TConfig>::MVector MVector;
        typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
        typedef typename TConfig::MemSpace memory_space;
        typedef typename TConfig::MatPrec  value_type;
        typedef typename TConfig::IndPrec  index_type;

        typedef typename TConfig::template setVecPrec<(AMGX_VecPrecision)AMGX_GET_MODE_VAL(AMGX_MatPrecision, TConfig::mode)>::Type vvec_value_type;
        typedef typename TConfig::template setVecPrec<AMGX_vecInt>::Type ivec_value_type;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;

        typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;

        typedef Vector<vvec_value_type> VVector;
        typedef Vector<ivec_value_type> IVector;
        typedef Vector<ivec_value_type_h> IVector_h;

        typedef Vector<i64vec_value_type> I64Vector;
        typedef Vector<i64vec_value_type_h> I64Vector_h;

        typedef typename ivec_value_type_h::VecPrec VecInt_t;

        inline DistributedArrangerBase(AMG_Config &cfg) : num_part(0), part_offsets(0), halo_coloring(FIRST) {halo_coloring = cfg.AMG_Config::getParameter<ColoringType>("halo_coloring", "default");}
        inline DistributedArrangerBase() : num_part(0), part_offsets(0), halo_coloring(FIRST) {};
        virtual ~DistributedArrangerBase() {};

        void set_part_offsets(const INDEX_TYPE num_part, const VecInt_t *part_offsets_h);
        void set_part_offsets(const INDEX_TYPE num_part, const int64_t *part_offsets_h);

        //virtual void create_B2L(Matrix<TConfig> &A, IVector &part_offsets, INDEX_TYPE num_part, INDEX_TYPE my_id, INDEX_TYPE rings, std::vector<Matrix<TConfig> > **matrix_halo) = 0;
        virtual void create_B2L(Matrix<TConfig> &A, INDEX_TYPE my_id, INDEX_TYPE rings) = 0;
        virtual void create_B2L(Matrix<TConfig> &A, INDEX_TYPE my_id, INDEX_TYPE rings, int64_t &base_index, INDEX_TYPE &index_range,
                                Vector<ivec_value_type_h> &neighbors, I64Vector &halo_ranges, std::vector<IVector > &B2L_maps, std::vector<IVector > &L2H_maps,
                                std::vector<std::vector<VecInt_t> > &B2L_rings, DistributedComms<TConfig> **comms,
                                std::vector<Matrix<TConfig> > **halo_rows, std::vector<DistributedManager<TConfig> > **halo_btl) = 0;
        virtual void create_B2L_from_neighbors(Matrix<TConfig>  &A,
                                               INDEX_TYPE my_id, INDEX_TYPE rings,
                                               int64_t base_index, INDEX_TYPE index_range,
                                               Vector<ivec_value_type_h> &neighbors,
                                               I64Vector_h &halo_ranges_h,
                                               I64Vector &halo_ranges,
                                               std::vector<IVector > &B2L_maps,
                                               std::vector<IVector > &L2H_maps,
                                               std::vector<std::vector<VecInt_t> > &B2L_rings,
                                               DistributedComms<TConfig> **comms,
                                               std::vector<Matrix<TConfig>  > **halo_rows,
                                               std::vector<DistributedManager<TConfig> > **halo_btl) = 0;

        virtual void create_B2L_from_maps(Matrix<TConfig>  &A,
                                          INDEX_TYPE my_id, INDEX_TYPE rings,
                                          int64_t base_index, INDEX_TYPE index_range,
                                          Vector<ivec_value_type_h> &neighbors,
                                          I64Vector &halo_ranges,
                                          std::vector<IVector > &B2L_maps,
                                          std::vector<std::vector<VecInt_t> > &B2L_rings,
                                          DistributedComms<TConfig> **comms,
                                          std::vector<Matrix<TConfig>  > **halo_rows,
                                          std::vector<DistributedManager<TConfig> > **halo_btl) = 0;

        virtual void create_B2L_from_maps(Matrix<TConfig>  &A,
                                          INDEX_TYPE my_id, INDEX_TYPE rings,
                                          Vector<ivec_value_type_h> &neighbors,
                                          std::vector<IVector > &B2L_maps,
                                          std::vector<IVector > &L2H_maps,
                                          std::vector<std::vector<VecInt_t> > &B2L_rings,
                                          DistributedComms<TConfig> **comms,
                                          std::vector<Matrix<TConfig>  > **halo_rows,
                                          std::vector<DistributedManager<TConfig> > **halo_btl) = 0;

        virtual void create_neighbors(Matrix<TConfig> &A, INDEX_TYPE my_id,
                                      int64_t &base_index, INDEX_TYPE &index_range,
                                      Vector<ivec_value_type_h> &neighbors,
                                      I64Vector_h &halo_ranges_h) = 0;

        virtual void create_neighbors(Matrix<TConfig> &A, INDEX_TYPE my_id,
                                      I64Vector_h &halo_ranges_h) = 0;

        virtual void append_neighbors(Matrix<TConfig> &A, IVector_h &neighbors, I64Vector_h &halo_ranges_h, I64Vector &halo_ranges, IVector_h &neighbor_flags, I64Vector_h &part_offsets_h) = 0;

        virtual void create_neighbors_v2(Matrix<TConfig> &A) = 0;

        virtual void create_neighbors_v2_global(Matrix<TConfig> &P, I64Vector &P_col_indices_global) = 0;

        virtual void update_neighbors_list(Matrix<TConfig> &A, IVector_h &neighbors, I64Vector_h &halo_ranges_h, I64Vector &halo_ranges, I64Vector_h &part_offsets_h, I64Vector &part_offsets, std::vector<IVector> &halo_rows_row_offsets, std::vector<I64Vector> &halo_rows_col_indices) = 0;

        virtual void create_B2L_one_ring(Matrix<TConfig> &A) = 0;
        virtual void createRowsLists(Matrix<TConfig> &A, bool is_matrix_P) = 0;

        virtual void create_one_ring_halo_rows(Matrix<TConfig> &A) = 0;

        virtual void create_boundary_lists(Matrix<TConfig> &A, int64_t base_index, INDEX_TYPE index_range,
                                           I64Vector_h &halo_ranges_h, I64Vector &halo_ranges,
                                           std::vector<IVector> &boundary_lists, IVector &halo_nodes) = 0;


        virtual  void create_boundary_lists(Matrix<TConfig> &A, I64Vector_h &halo_ranges_h,
                                            std::vector<IVector> &boundary_lists, IVector &halo_nodes) = 0;

        virtual void create_boundary_lists_v2(Matrix<TConfig> &A,
                                              std::vector<IVector> &boundary_lists) = 0;

        virtual void initialize_B2L_maps_offsets(Matrix<TConfig> &A, int num_rings) = 0;

        virtual void create_boundary_lists_v3(Matrix<TConfig> &A) = 0;

        virtual void compute_local_halo_indices(IVector &A_row_offsets,
                                                IVector &A_col_indices,
                                                std::vector<IVector> &halo_row_offsets,
                                                std::vector<I64Vector> &halo_global_indices,
                                                std::vector<IVector> &halo_local_indices,
                                                I64Vector &local_to_global,
                                                std::vector<IVector> &boundary_lists,
                                                IVector_h &neighbors,
                                                I64Vector_h &halo_ranges_h,
                                                I64Vector &halo_ranges,
                                                IVector_h &halo_offsets,
                                                int64_t base_index,
                                                int64_t index_range,
                                                int A_num_rows,
                                                int current_num_rings) = 0;

        virtual void renumber_to_local(Matrix<TConfig> &A,
                                       std::vector<IVector> &boundary_lists, std::vector<IVector *> &halo_lists,
                                       INDEX_TYPE my_id,
                                       int64_t base_index, INDEX_TYPE index_range,
                                       Vector<ivec_value_type_h> &neighbors,
                                       I64Vector_h &halo_ranges_h,
                                       I64Vector &halo_ranges,
                                       IVector &halo_nodes) = 0;

        virtual void renumber_to_local(Matrix<TConfig> &A,
                                       std::vector<IVector> &boundary_lists, std::vector<IVector *> &halo_lists,
                                       INDEX_TYPE my_id,
                                       I64Vector_h &halo_ranges_h, IVector &halo_nodes) = 0;

        virtual void create_rings(Matrix<TConfig> &A, INDEX_TYPE rings,
                                  INDEX_TYPE num_neighbors,
                                  std::vector<IVector > &B2L_maps,
                                  std::vector<std::vector<VecInt_t> > &B2L_rings) = 0;

        virtual void create_rings(Matrix<TConfig> &A, INDEX_TYPE rings) = 0;

        virtual void create_maps(Matrix<TConfig> &A, INDEX_TYPE rings,
                                 INDEX_TYPE num_neighbors,
                                 std::vector<IVector > &B2L_maps,
                                 std::vector<std::vector<VecInt_t> > &B2L_rings,
                                 std::vector<IVector> &boundary_lists) = 0;

        virtual void create_maps(Matrix<TConfig> &A, INDEX_TYPE rings,
                                 std::vector<IVector> &boundary_lists) = 0;

        virtual void create_halo_matrices(Matrix<TConfig> &A, INDEX_TYPE rings,
                                          int64_t base_index, INDEX_TYPE index_range,
                                          INDEX_TYPE global_id, Vector<ivec_value_type_h> &neighbors,
                                          std::vector<IVector > &B2L_maps,
                                          std::vector<std::vector<VecInt_t> > &B2L_rings,
                                          std::vector<Matrix<TConfig> > **halo_rows,
                                          std::vector<DistributedManager<TConfig> > **halo_btl,
                                          std::vector<IVector> &boundary_lists,
                                          std::vector<IVector *> &halo_lists) = 0;
        virtual void create_halo_matrices(Matrix<TConfig> &A, INDEX_TYPE rings, std::vector<IVector> &boundary_lists, std::vector<IVector *> &halo_lists) = 0;

        virtual void create_halo_rows(Matrix<TConfig> &A, INDEX_TYPE rings, INDEX_TYPE num_neighbors,
                                      std::vector<IVector > &B2L_maps, std::vector<std::vector<VecInt_t> > &B2L_rings,
                                      std::vector<Matrix<TConfig> > **halo_rows, std::vector<IVector *> &halo_lists) = 0;



        virtual void exchange_halo_rows_P(Matrix<TConfig> &A, Matrix<TConfig> &P, I64Vector &local_to_global_map, IVector_h &P_neighbors, I64Vector_h &P_halo_ranges_h, I64Vector &P_halo_ranges, IVector_h &P_halo_offsets, I64Vector_h &RAP_part_offsets_h, I64Vector &RAP_part_offsets, index_type num_owned_coarse_pts, int64_t base_index) = 0;

        virtual void append_halo_rows(Matrix<TConfig> &A, std::vector<IVector> &halo_row_offsets, std::vector<IVector> &halo_row_local_indices, std::vector<MVector> &halo_row_values ) = 0;

        virtual void exchange_RAP_ext(Matrix<TConfig> &RAP, Matrix<TConfig> &RAP_full, Matrix<TConfig> &A, Matrix<TConfig> &P, IVector_h &P_halo_offsets_h, I64Vector &P_local_to_global_map, IVector_h &P_neighbors, I64Vector_h &P_halo_ranges_h, I64Vector &P_halo_ranges, I64Vector_h &RAP_part_offsets_h, I64Vector &RAP_part_offsets, index_type num_owned_coarse_pts, int64_t coarse_base_index, void *wk) = 0;

        virtual void pack_halo_rows_P(Matrix<TConfig> &A, Matrix<TConfig> &P, std::vector<IVector> &halo_rows_P_row_offsets, std::vector<I64Vector> &halo_rows_P_col_indices, std::vector<MVector> &halo_rows_P_values, I64Vector &P_local_to_global_map, index_type num_owned_coarse_pts, int64_t coarse_base_index) = 0;

        virtual void pack_halo_rows_RAP(Matrix<TConfig> &RAP, Matrix<TConfig> &P, std::vector<IVector> &halo_rows_RAP_row_offsets, std::vector<I64Vector> &halo_rows_RAP_col_indices, std::vector<MVector> &halo_rows_RAP_values, std::vector<I64Vector> &halo_rows_RAP_row_ids, I64Vector &RAP_local_to_global_map) = 0;

        virtual void sparse_matrix_add(Matrix<TConfig> &RAP, std::vector<IVector> &halo_rows_RAP_row_offsets, std::vector<I64Vector> &halo_rows_RAP_col_indices, std::vector<MVector> &halo_rows_RAP_values, std::vector<I64Vector> halo_rows_RAP_row_ids, I64Vector &P_local_to_global_map, index_type P_col_num_owned_pts, int64_t P_col_base_index) = 0;

        virtual void initialize_manager(Matrix<TConfig> &A,
                                        Matrix<TConfig> &P,
                                        int num_owned_coarse_pts) = 0;

        virtual void initialize_manager_from_global_col_indices(Matrix<TConfig> &P,
                I64Vector &P_col_indices_global) = 0;

        virtual void createTempManager( Matrix<TConfig> &B, Matrix<TConfig> &A, IVector &offsets) = 0;

        virtual void create_halo_rows(Matrix<TConfig> &A, std::vector<IVector *> &halo_lists) = 0;
        virtual void create_halo_rows_global_indices(Matrix<TConfig> &A, std::vector<IVector> &halo_rows_row_offsets, std::vector<I64Vector> &halo_rows_col_indices, std::vector<MVector> &halo_rows_values) = 0;

        virtual void create_halo_btl(Matrix<TConfig> &A, INDEX_TYPE rings, INDEX_TYPE num_neighbors,
                                     Vector<ivec_value_type_h> &neighbors_list,
                                     int64_t base_index, INDEX_TYPE index_range,
                                     INDEX_TYPE global_id, std::vector<IVector > &B2L_maps,
                                     std::vector<std::vector<VecInt_t> > &B2L_rings,
                                     std::vector<DistributedManager<TConfig> > **halo_btl,
                                     std::vector<IVector *> &halo_lists
                                    ) = 0;
        virtual void create_halo_btl(Matrix<TConfig> &A, std::vector<IVector *> &halo_lists) = 0;
        virtual void create_halo_btl_multiple_rings(Matrix<TConfig> &A, int rings) = 0;

    protected:
        INDEX_TYPE num_part;
        I64Vector part_offsets;
        ColoringType halo_coloring;
};

// specialization for host
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class DistributedArranger< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > : public DistributedArrangerBase< TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef typename Matrix<TConfig>::MVector MVector;
        typedef typename TConfig::template setMemSpace<AMGX_host>::Type TConfig_h;
        typedef typename TConfig::template setMemSpace<AMGX_device>::Type TConfig_d;
        typedef DistributedManager<TConfig_h> Manager_h;
        typedef DistributedManager<TConfig_d> Manager_d;
        typedef typename TConfig::MemSpace memory_space;
        typedef typename TConfig::MatPrec  value_type;
        typedef typename TConfig::IndPrec  index_type;
        typedef Matrix<TConfig_h> Matrix_h;
        typedef typename DistributedManagerBase<TConfig>::VVector VVector;
        typedef typename DistributedManagerBase<TConfig>::IVector IVector;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
        typedef typename ivec_value_type_h::VecPrec VecInt_t;

        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;
        typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type;

        typedef Vector<ivec_value_type_h> IVector_h;
        typedef Vector<i64vec_value_type_h> I64Vector_h;
        typedef Vector<i64vec_value_type_h> I64Vector;

        //constructors
        inline DistributedArranger() : DistributedArrangerBase<TConfig>() {}
        inline DistributedArranger(AMG_Config &cfg) : DistributedArrangerBase<TConfig>(cfg) {}

        //destructor
        virtual ~DistributedArranger() {}

        void create_B2L(Matrix_h &A, INDEX_TYPE my_id, INDEX_TYPE rings);
        void create_B2L(Matrix_h &A, INDEX_TYPE my_id, INDEX_TYPE rings, int64_t &base_index, INDEX_TYPE &index_range,
                        Vector<ivec_value_type_h> &neighbors, I64Vector &halo_ranges, std::vector<IVector > &B2L_maps, std::vector<IVector > &L2H_maps,
                        std::vector<std::vector<VecInt_t> > &B2L_rings, DistributedComms<TConfig_h> **comms,
                        std::vector<Matrix_h > **halo_rows,
                        std::vector<Manager_h > **halo_btl) {};

        void create_B2L_from_neighbors(Matrix_h &A,
                                       INDEX_TYPE my_id, INDEX_TYPE rings,
                                       int64_t base_index, INDEX_TYPE index_range,
                                       Vector<ivec_value_type_h> &neighbors,
                                       I64Vector_h &halo_ranges_h,
                                       I64Vector &halo_ranges,
                                       std::vector<IVector > &B2L_maps,
                                       std::vector<IVector > &L2H_maps,
                                       std::vector<std::vector<VecInt_t> > &B2L_rings,
                                       DistributedComms<TConfig_h> **comms,
                                       std::vector<Matrix_h > **halo_rows,
                                       std::vector<Manager_h > **halo_btl) {};

        void create_B2L_from_maps(Matrix_h &A,
                                  INDEX_TYPE my_id, INDEX_TYPE rings,
                                  int64_t base_index, INDEX_TYPE index_range,
                                  Vector<ivec_value_type_h> &neighbors,
                                  I64Vector &halo_ranges,
                                  std::vector<IVector > &B2L_maps,
                                  std::vector<std::vector<VecInt_t> > &B2L_rings,
                                  DistributedComms<TConfig_h> **comms,
                                  std::vector<Matrix_h > **halo_rows,
                                  std::vector<Manager_h > **halo_btl) {};

        void create_B2L_from_maps(Matrix_h &A,
                                  INDEX_TYPE my_id, INDEX_TYPE rings,
                                  Vector<ivec_value_type_h> &neighbors,
                                  std::vector<IVector > &B2L_maps,
                                  std::vector<IVector > &L2H_maps,
                                  std::vector<std::vector<VecInt_t> > &B2L_rings,
                                  DistributedComms<TConfig_h> **comms,
                                  std::vector<Matrix_h > **halo_rows,
                                  std::vector<Manager_h > **halo_btl) {};

        void create_neighbors(Matrix_h &A, INDEX_TYPE my_id,
                              int64_t &base_index, INDEX_TYPE &index_range,
                              Vector<ivec_value_type_h> &neighbors,
                              I64Vector_h &halo_ranges_h) {};

        void create_neighbors(Matrix_h &A, INDEX_TYPE my_id, I64Vector_h &halo_ranges_h) {};

        void append_neighbors(Matrix<TConfig> &A, IVector_h &neighbors, I64Vector_h &halo_ranges_h, I64Vector &halo_ranges, IVector_h &neighbor_flags, I64Vector_h &part_offsets_h) {};

        void create_neighbors_v2(Matrix<TConfig> &A) {};

        void create_neighbors_v2_global(Matrix_h &P, I64Vector_h &P_col_indices_global) {};

        void update_neighbors_list(Matrix<TConfig> &A, IVector_h &neighbors, I64Vector_h &halo_ranges_h, I64Vector &halo_ranges, I64Vector_h &part_offsets_h, I64Vector &part_offsets, std::vector<IVector> &halo_rows_row_offsets, std::vector<I64Vector> &halo_rows_col_indices) {};

        void create_B2L_one_ring(Matrix<TConfig> &A) {};
        void createRowsLists(Matrix<TConfig> &A, bool is_matrix_P) {};

        void create_one_ring_halo_rows(Matrix<TConfig> &A) {};

        void create_boundary_lists(Matrix_h &A, int64_t base_index, INDEX_TYPE index_range,
                                   I64Vector_h &halo_ranges_h, I64Vector &halo_ranges,
                                   std::vector<IVector> &boundary_lists, IVector &halo_nodes) {};

        void create_boundary_lists(Matrix_h &A, I64Vector_h &halo_ranges_h,
                                   std::vector<IVector> &boundary_lists, IVector &halo_nodes) {};

        void create_boundary_lists_v2(Matrix_h &A,
                                      std::vector<IVector> &boundary_lists) {};

        void initialize_B2L_maps_offsets(Matrix_h &A, int num_rings) {};

        void create_boundary_lists_v3(Matrix_h &A) {};

        void compute_local_halo_indices(IVector &A_row_offsets,
                                        IVector &A_col_indices,
                                        std::vector<IVector> &halo_row_offsets,
                                        std::vector<I64Vector> &halo_global_indices,
                                        std::vector<IVector> &halo_local_indices,
                                        I64Vector &local_to_global,
                                        std::vector<IVector> &boundary_lists,
                                        IVector_h &neighbors,
                                        I64Vector_h &halo_ranges_h,
                                        I64Vector &halo_ranges,
                                        IVector_h &halo_offsets,
                                        int64_t base_index,
                                        int64_t index_range,
                                        int A_num_rows,
                                        int current_num_rings) {};


        void create_rings(Matrix_h &A, INDEX_TYPE rings,
                          INDEX_TYPE num_neighbors,
                          std::vector<IVector > &B2L_maps,
                          std::vector<std::vector<VecInt_t> > &B2L_rings) {};
        void create_rings(Matrix_h &A, INDEX_TYPE rings) {};

        void create_maps(Matrix_h &A, INDEX_TYPE rings,
                         INDEX_TYPE num_neighbors,
                         std::vector<IVector > &B2L_maps,
                         std::vector<std::vector<VecInt_t> > &B2L_rings, std::vector<IVector> &boundary_lists) {};
        void create_maps(Matrix_h &A, INDEX_TYPE rings, std::vector<IVector> &boundary_lists) {};

        void create_halo_matrices(Matrix_h &A, INDEX_TYPE rings, std::vector<IVector> &boundary_lists, std::vector<IVector *> &halo_lists) {};
        void create_halo_matrices(Matrix_h &A, INDEX_TYPE rings,
                                  int64_t base_index, INDEX_TYPE index_range,
                                  INDEX_TYPE global_id, Vector<ivec_value_type_h> &neighbors,
                                  std::vector<IVector > &B2L_maps,
                                  std::vector<std::vector<VecInt_t> > &B2L_rings,
                                  std::vector<Matrix_h > **halo_rows,
                                  std::vector<Manager_h > **halo_btl,
                                  std::vector<IVector> &boundary_lists,
                                  std::vector<IVector *> &halo_lists) {};

        void create_halo_rows(Matrix_h &A, INDEX_TYPE rings, INDEX_TYPE num_neighbors,
                              std::vector<IVector > &B2L_maps, std::vector<std::vector<VecInt_t> > &B2L_rings,
                              std::vector<Matrix<TConfig> > **halo_rows,
                              std::vector<IVector *> &halo_lists) {};


        void exchange_halo_rows_P(Matrix_h &A, Matrix_h &P, I64Vector &local_to_global_map, IVector_h &P_neighbors, I64Vector_h &P_halo_ranges_h, I64Vector &P_halo_ranges, IVector_h &P_halo_offsets, I64Vector_h &RAP_part_offsets_h, I64Vector &RAP_part_offsets, index_type num_owned_coarse_pts, int64_t base_index) {};

        void append_halo_rows(Matrix<TConfig> &A, std::vector<IVector> &halo_row_offsets, std::vector<IVector> &halo_row_local_indices, std::vector<MVector> &halo_row_values ) {};

        void exchange_RAP_ext(Matrix_h &RAP, Matrix_h &RAP_full, Matrix_h &A, Matrix_h &P, IVector_h &P_halo_offsets_h, I64Vector &P_local_to_global_map, IVector_h &P_neighbors, I64Vector_h &P_halo_ranges_h, I64Vector &P_halo_ranges, I64Vector_h &RAP_part_offsets_h, I64Vector &RAP_part_offsets, index_type num_owned_coarse_pts, int64_t coarse_base_index, void *wk) {};

        void pack_halo_rows_P(Matrix_h &A, Matrix_h &P, std::vector<IVector> &halo_rows_P_row_offsets, std::vector<I64Vector> &halo_rows_P_col_indices, std::vector<MVector> &halo_rows_P_values, I64Vector &P_local_to_global_map, index_type num_owned_coarse_pts, int64_t coarse_base_index) {};

        void pack_halo_rows_RAP(Matrix_h &RAP, Matrix_h &P, std::vector<IVector> &halo_rows_RAP_row_offsets, std::vector<I64Vector> &halo_rows_RAP_col_indices, std::vector<MVector> &halo_rows_RAP_values, std::vector<I64Vector> &halo_rows_RAP_row_ids, I64Vector &RAP_local_to_global_map) {};

        void sparse_matrix_add(Matrix_h &RAP, std::vector<IVector> &halo_rows_RAP_row_offsets, std::vector<I64Vector> &halo_rows_RAP_col_indices, std::vector<MVector> &halo_rows_RAP_values, std::vector<I64Vector> halo_rows_RAP_row_ids, I64Vector &P_local_to_global_map, index_type P_col_num_owned_pts, int64_t P_col_base_index) {};

        void initialize_manager(Matrix_h &A,
                                Matrix_h &P,
                                int num_owned_coarse_pts) {};

        void initialize_manager_from_global_col_indices(Matrix_h &P,
                I64Vector_h &P_col_indices_global) {};


        void createTempManager( Matrix_h &B, Matrix_h &A, IVector &offsets) {};

        void create_halo_rows(Matrix_h &A, std::vector<IVector *> &halo_lists) {};
        void create_halo_rows_global_indices(Matrix_h &A, std::vector<IVector> &halo_rows_row_offsets, std::vector<I64Vector> &halo_rows_col_indices, std::vector<MVector> &halo_rows_values) {};

        void create_halo_btl(Matrix_h &A, INDEX_TYPE rings, INDEX_TYPE num_neighbors,
                             Vector<ivec_value_type_h> &neighbors_list,
                             int64_t base_index, INDEX_TYPE index_range,
                             INDEX_TYPE global_id, std::vector<IVector > &B2L_maps,
                             std::vector<std::vector<VecInt_t> > &B2L_rings,
                             std::vector<Manager_h > **halo_btl,
                             std::vector<IVector *> &halo_lists
                            ) {};
        /*void create_halo_btl(Matrix_h &A, INDEX_TYPE rings,
                             INDEX_TYPE base_index, INDEX_TYPE index_range, INDEX_TYPE num_neighbors,
                             Vector<ivec_value_type_h>& neighbors_list,
                             INDEX_TYPE global_id, std::vector<IVector >& B2L_maps,
                             std::vector<std::vector<VecInt_t> >& B2L_rings,
                             std::vector<Manager_h > **halo_btl){};*/
        void create_halo_btl(Matrix_h &A, std::vector<IVector *> &halo_lists) {};
        void create_halo_btl_multiple_rings(Matrix_h &A, int rings) {};

        void renumber_to_local(Matrix_h &A,
                               std::vector<IVector> &boundary_lists, std::vector<IVector *> &halo_lists,
                               INDEX_TYPE my_id,
                               int64_t base_index, INDEX_TYPE index_range,
                               Vector<ivec_value_type_h> &neighbors,
                               I64Vector_h &halo_ranges_h,
                               I64Vector &halo_ranges,
                               IVector &halo_nodes) {};
        void renumber_to_local(Matrix_h &A,
                               std::vector<IVector> &boundary_lists, std::vector<IVector *> &halo_lists,
                               INDEX_TYPE my_id, I64Vector_h &halo_ranges_h, IVector &halo_nodes) {};
        void prepare_local(Matrix_h &A,
                           INDEX_TYPE my_id, INDEX_TYPE rings,
                           int64_t base_index, INDEX_TYPE index_range,
                           Vector<ivec_value_type_h> &neighbors,
                           I64Vector_h &halo_ranges_h,
                           I64Vector &halo_ranges,
                           std::vector<IVector > &L2H_maps,
                           std::vector<IVector> &boundary_lists,
                           std::vector<IVector *> &halo_lists,
                           DistributedComms<TConfig_h> **comms) {};
};

// specialization for device
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class DistributedArranger< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > : public DistributedArrangerBase< TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
    public:
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig;
        typedef typename Matrix<TConfig>::MVector MVector;
        typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
        typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
        typedef DistributedManager<TConfig_h> Manager_h;
        typedef DistributedManager<TConfig_d> Manager_d;
        typedef typename TConfig::IndPrec  index_type;
        typedef typename DistributedManagerBase<TConfig>::VVector VVector;
        typedef typename DistributedManagerBase<TConfig>::IVector IVector;
        typedef Matrix<TConfig_d> Matrix_d;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
        typedef typename DistributedManagerBase<TConfig_h>::IVector IVector_h;
        typedef typename ivec_value_type_h::VecPrec VecInt_t;

        typedef typename TConfig::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type;
        typedef typename TConfig_d::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_d;
        typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;

        typedef Vector<i64vec_value_type> I64Vector;
        typedef Vector<i64vec_value_type_h> I64Vector_h;
        typedef Vector<i64vec_value_type_d> I64Vector_d;

        //constructors
        inline DistributedArranger() : DistributedArrangerBase<TConfig>() {}
        inline DistributedArranger(AMG_Config &cfg) : DistributedArrangerBase<TConfig>(cfg) {}

        //destructor
        virtual ~DistributedArranger() {}

        //std::vector<IVector> bak;

        void create_B2L(Matrix_d &A, INDEX_TYPE my_id, INDEX_TYPE rings);
        void create_B2L(Matrix_d &A, INDEX_TYPE my_id, INDEX_TYPE rings, int64_t &base_index, INDEX_TYPE &index_range,
                        Vector<ivec_value_type_h> &neighbors, I64Vector_d &halo_ranges, std::vector<IVector > &B2L_maps, std::vector<IVector > &L2H_maps,
                        std::vector<std::vector<VecInt_t> > &B2L_rings, DistributedComms<TConfig_d> **comms,
                        std::vector<Matrix_d > **halo_rows,
                        std::vector<Manager_d > **halo_btl);
        void create_B2L_from_neighbors(Matrix_d &A,
                                       INDEX_TYPE my_id, INDEX_TYPE rings,
                                       int64_t base_index, INDEX_TYPE index_range,
                                       Vector<ivec_value_type_h> &neighbors,
                                       I64Vector_h &halo_ranges_h,
                                       I64Vector_d &halo_ranges,
                                       std::vector<IVector > &B2L_maps,
                                       std::vector<IVector > &L2H_maps,
                                       std::vector<std::vector<VecInt_t> > &B2L_rings,
                                       DistributedComms<TConfig_d> **comms,
                                       std::vector<Matrix_d > **halo_rows,
                                       std::vector<Manager_d > **halo_btl);

        void create_B2L_from_maps(Matrix_d &A,
                                  INDEX_TYPE my_id, INDEX_TYPE rings,
                                  int64_t base_index, INDEX_TYPE index_range,
                                  Vector<ivec_value_type_h> &neighbors,
                                  I64Vector_d &halo_ranges,
                                  std::vector<IVector > &B2L_maps,
                                  std::vector<std::vector<VecInt_t> > &B2L_rings,
                                  DistributedComms<TConfig_d> **comms,
                                  std::vector<Matrix_d > **halo_rows,
                                  std::vector<Manager_d > **halo_btl);

        void create_B2L_from_maps(Matrix_d &A,
                                  INDEX_TYPE my_id, INDEX_TYPE rings,
                                  Vector<ivec_value_type_h> &neighbors,
                                  std::vector<IVector > &B2L_maps,
                                  std::vector<IVector > &L2H_maps,
                                  std::vector<std::vector<VecInt_t> > &B2L_rings,
                                  DistributedComms<TConfig_d> **comms,
                                  std::vector<Matrix_d > **halo_rows,
                                  std::vector<Manager_d > **halo_btl);

        void create_neighbors(Matrix_d &A, INDEX_TYPE my_id,
                              int64_t &base_index, INDEX_TYPE &index_range,
                              Vector<ivec_value_type_h> &neighbors,
                              I64Vector_h &halo_ranges_h);

        void create_neighbors(Matrix_d &A, INDEX_TYPE my_id,
                              I64Vector_h &halo_ranges_h);

        void append_neighbors(Matrix<TConfig> &A, IVector_h &neighbors, I64Vector_h &halo_ranges_h, I64Vector &halo_ranges, IVector_h &neighbor_flags, I64Vector_h &part_offsets_h);

        void create_neighbors_v2(Matrix<TConfig> &A);

        void create_neighbors_v2_global(Matrix_d &P, I64Vector_d &P_col_indices_global);

        void update_neighbors_list(Matrix<TConfig> &A, IVector_h &neighbors, I64Vector_h &halo_ranges_h, I64Vector &halo_ranges, I64Vector_h &part_offsets_h, I64Vector &part_offsets, std::vector<IVector> &halo_rows_row_offsets, std::vector<I64Vector> &halo_rows_col_indices);

        void create_B2L_one_ring(Matrix<TConfig> &A);
        void createRowsLists(Matrix<TConfig> &A, bool is_matrix_P);

        void create_one_ring_halo_rows(Matrix<TConfig> &A);

        void create_boundary_lists(Matrix_d &A, int64_t base_index, INDEX_TYPE index_range,
                                   I64Vector_h &halo_ranges_h, I64Vector_d &halo_ranges,
                                   std::vector<IVector> &boundary_lists, IVector &halo_nodes);

        void create_boundary_lists(Matrix_d &A, I64Vector_h &halo_ranges_h,
                                   std::vector<IVector> &boundary_lists, IVector &halo_nodes);

        void create_boundary_lists_v2(Matrix_d &A, std::vector<IVector> &boundary_lists);

        void create_boundary_lists_v3(Matrix_d &A);

        void initialize_B2L_maps_offsets(Matrix_d &A, int num_rings);

        void compute_local_halo_indices(IVector &A_row_offsets,
                                        IVector &A_col_indices,
                                        std::vector<IVector> &halo_row_offsets,
                                        std::vector<I64Vector> &halo_global_indices,
                                        std::vector<IVector> &halo_local_indices,
                                        I64Vector &local_to_global,
                                        std::vector<IVector> &boundary_lists,
                                        IVector_h &neighbors,
                                        I64Vector_h &halo_ranges_h,
                                        I64Vector &halo_ranges,
                                        IVector_h &halo_offsets,
                                        int64_t base_index,
                                        int64_t index_range,
                                        int A_num_rows,
                                        int current_num_rings);

        void create_rings(Matrix_d &A, INDEX_TYPE rings,
                          INDEX_TYPE num_neighbors,
                          std::vector<IVector > &B2L_maps,
                          std::vector<std::vector<VecInt_t> > &B2L_rings);
        void create_rings(Matrix_d &A, INDEX_TYPE rings);

        void create_maps(Matrix_d &A, INDEX_TYPE rings,
                         INDEX_TYPE num_neighbors,
                         std::vector<IVector > &B2L_maps,
                         std::vector<std::vector<VecInt_t> > &B2L_rings,
                         std::vector<IVector> &boundary_lists);
        void create_maps(Matrix_d &A, INDEX_TYPE rings, std::vector<IVector> &boundary_lists);

        void create_halo_matrices(Matrix_d &A, INDEX_TYPE rings, std::vector<IVector> &boundary_lists, std::vector<IVector *> &halo_lists);
        void create_halo_matrices(Matrix_d &A, INDEX_TYPE rings,
                                  int64_t base_index, INDEX_TYPE index_range,
                                  INDEX_TYPE global_id, Vector<ivec_value_type_h> &neighbors,
                                  std::vector<IVector > &B2L_maps,
                                  std::vector<std::vector<VecInt_t> > &B2L_rings,
                                  std::vector<Matrix_d > **halo_rows,
                                  std::vector<Manager_d > **halo_btl,
                                  std::vector<IVector> &boundary_lists,
                                  std::vector<IVector *> &halo_lists);

        void create_halo_rows( Matrix_d &A, INDEX_TYPE rings, INDEX_TYPE num_neighbors,
                               std::vector<IVector > &B2L_maps,
                               std::vector<std::vector<VecInt_t> > &B2L_rings,
                               std::vector<Matrix<TConfig> > **halo_rows,
                               std::vector<IVector *> &halo_lists);


        void exchange_halo_rows_P(Matrix_d &A, Matrix_d &P, I64Vector &local_to_global_map, IVector_h &P_neighbors, I64Vector_h &P_halo_ranges_h, I64Vector &P_halo_ranges, IVector_h &P_halo_offsets, I64Vector_h &RAP_part_offsets_h, I64Vector &RAP_part_offsets, index_type num_owned_coarse_pts, int64_t base_index);

        void append_halo_rows(Matrix<TConfig> &A, std::vector<IVector> &halo_row_offsets, std::vector<IVector> &halo_row_local_indices, std::vector<MVector> &halo_row_values) ;

        void exchange_RAP_ext(Matrix_d &RAP, Matrix_d &RAP_full, Matrix_d &A, Matrix_d &P, IVector_h &P_halo_offsets_h, I64Vector &P_local_to_global_map, IVector_h &P_neighbors, I64Vector_h &P_halo_ranges_h, I64Vector &P_halo_ranges, I64Vector_h &RAP_part_offsets_h, I64Vector &RAP_part_offsets, index_type num_owned_coarse_pts, int64_t coarse_base_index, void *wk);

        void pack_halo_rows_P(Matrix_d &A, Matrix_d &P, std::vector<IVector> &halo_rows_P_row_offsets, std::vector<I64Vector> &halo_rows_P_col_indices, std::vector<MVector> &halo_rows_P_values, I64Vector &P_local_to_global_map, index_type num_owned_coarse_pts, int64_t coarse_base_index);

        void pack_halo_rows_RAP(Matrix_d &RAP, Matrix_d &P, std::vector<IVector> &halo_rows_RAP_row_offsets, std::vector<I64Vector> &halo_rows_RAP_col_indices, std::vector<MVector> &halo_rows_RAP_values, std::vector<I64Vector> &halo_rows_RAP_row_ids, I64Vector &RAP_local_to_global_map);

        void sparse_matrix_add(Matrix_d &RAP, std::vector<IVector> &halo_rows_RAP_row_offsets, std::vector<I64Vector> &halo_rows_RAP_col_indices, std::vector<MVector> &halo_rows_RAP_values, std::vector<I64Vector> halo_rows_RAP_row_ids, I64Vector &P_local_to_global_map, index_type P_col_num_owned_pts, int64_t P_col_base_index);

        void initialize_manager(Matrix_d &A,
                                Matrix_d &P,
                                int num_owned_coarse_pts);

        void initialize_manager_from_global_col_indices(Matrix_d &P,
                I64Vector_d &P_col_indices_global);

        void createTempManager( Matrix_d &B, Matrix_d &A, IVector &offsets);

        void create_halo_rows( Matrix_d &A, std::vector<IVector *> &halo_lists);
        void create_halo_rows_global_indices( Matrix_d &A, std::vector<IVector> &halo_rows_row_offsets, std::vector<I64Vector> &halo_rows_col_indices, std::vector<MVector> &halo_rows_values);

        void create_halo_btl( Matrix_d &A, INDEX_TYPE rings, INDEX_TYPE num_neighbors,
                              Vector<ivec_value_type_h> &neighbors_list,
                              int64_t base_index, INDEX_TYPE index_range,
                              INDEX_TYPE global_id, std::vector<IVector > &B2L_maps,
                              std::vector<std::vector<VecInt_t> > &B2L_rings,
                              std::vector<Manager_d > **halo_btl,
                              std::vector<IVector *> &halo_lists);
        void create_halo_btl( Matrix_d &A, std::vector<IVector *> &halo_lists);
        void create_halo_btl_multiple_rings(Matrix_d &A, int rings);

        void renumber_to_local(Matrix_d &A,
                               std::vector<IVector> &boundary_lists, std::vector<IVector *> &halo_lists,
                               INDEX_TYPE my_id,
                               int64_t base_index, INDEX_TYPE index_range,
                               Vector<ivec_value_type_h> &neighbors,
                               I64Vector_h &halo_ranges_h,
                               I64Vector_d &halo_ranges,
                               IVector &halo_nodes);
        void renumber_to_local(Matrix_d &A,
                               std::vector<IVector> &boundary_lists, std::vector<IVector *> &halo_lists,
                               INDEX_TYPE my_id, I64Vector_h &halo_ranges_h, IVector &halo_nodes);
        void prepare_local(Matrix_d &A,
                           INDEX_TYPE my_id, INDEX_TYPE rings,
                           int64_t base_index, INDEX_TYPE index_range,
                           Vector<ivec_value_type_h> &neighbors,
                           I64Vector_h &halo_ranges_h,
                           I64Vector_d &halo_ranges,
                           std::vector<IVector > &L2H_maps,
                           std::vector<IVector> &boundary_lists,
                           std::vector<IVector *> &halo_lists,
                           DistributedComms<TConfig_d> **comms);
    private:
        //std::vector<IVector> boundary_lists;
        //std::vector<IVector *> halo_lists;
        //IVector *halo_nodes_p;
};
}


