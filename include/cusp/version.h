// SPDX-FileCopyrightText: 2008 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file version.h
 *  \brief Cusp version
 */

#pragma once

#include <cusp/detail/config.h>

//  This is the only cusp header that is guaranteed to 
//  change with every cusp release.
//
//  CUSP_VERSION % 100 is the sub-minor version
//  CUSP_VERSION / 100 % 1000 is the minor version
//  CUSP_VERSION / 100000 is the major version

#define CUSP_VERSION 300
#define CUSP_MAJOR_VERSION     (CUSP_VERSION / 100000)
#define CUSP_MINOR_VERSION     (CUSP_VERSION / 100 % 1000)
#define CUSP_SUBMINOR_VERSION  (CUSP_VERSION % 100)

