// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace amgx
{

//computes B=A^T
template <class Matrix>
void transpose(const Matrix &A, Matrix &B);

template <class Matrix>
void transpose(const Matrix &A, Matrix &B, int num_rows);

} // namespace amgx
