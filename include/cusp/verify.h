/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file verify.h
 *  \brief Validate matrix format
 */

#pragma once

#include <cusp/detail/config.h>

namespace cusp
{

template <typename MatrixType>
bool is_valid_matrix(const MatrixType& A);

template <typename MatrixType, typename OutputStream>
bool is_valid_matrix(const MatrixType& A, OutputStream& ostream);

template <typename MatrixType>
void assert_is_valid_matrix(const MatrixType& A);

} // end namespace cusp

#include <cusp/detail/verify.inl>

