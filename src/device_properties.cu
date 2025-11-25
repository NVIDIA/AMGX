// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <device_properties.h>
#include <error.h>
namespace amgx
{
static cudaDeviceProp deviceProps;
static bool initialized=false;

cudaDeviceProp getDeviceProperties()
{
    if(!initialized) {
        int dev;
        cudaGetDevice(&dev);
        cudaCheckError();
        cudaGetDeviceProperties(&deviceProps, dev);
        cudaCheckError();
        initialized=true;
    }
    return deviceProps;
}

// Return the number of Streaming Multiprocessors on the current device
int getSMCount()
{
    auto devProp = getDeviceProperties();
    return devProp.multiProcessorCount;
}

}
