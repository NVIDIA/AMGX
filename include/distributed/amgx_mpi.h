// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#ifdef AMGX_WITH_MPI
#include <mpi.h>

namespace amgx
{
void installMPIErrorHandler(MPI_Comm comm);
void uninstallMPIErrorHandler(MPI_Comm comm);
}
#else

#endif


