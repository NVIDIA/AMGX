// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __ELL_H__
#define __ELL_H__

////////////////////////////////////////////////////////////////////////////////
//! Compute y += A*x for a sparse ELL matrix A and column vectors x and y
//! @param num_rows             number of rows in A
//! @param num_cols             number of columns in A
//! @param num_entries_per_row  number columns in each row (smaller rows are zero padded)
//! @param stride               seperation between row entries (stride >= num_rows, for alignment)
//! @param Aj                   ELL column indices
//! @param Ax                   ELL nonzero values
//! @param x                    column vector
//! @param y                    column vector
////////////////////////////////////////////////////////////////////////////////
template <typename IndexType, typename ValueType>
void ell_matvec(const IndexType num_rows,
                const IndexType num_cols,
                const IndexType num_entries_per_row,
                const IndexType stride,
                const IndexType * Aj, 
                const ValueType * Ax, 
                const ValueType * x,
                      ValueType * y)
{
    for(IndexType n = 0; n < num_entries_per_row; n++){
        const IndexType * Aj_n = Aj + n * stride;
        const ValueType * Ax_n = Ax + n * stride;
        for(IndexType i = 0; i < num_rows; i++){
            y[i] += Ax_n[i] * x[Aj_n[i]];
        }
    }
}



#endif
