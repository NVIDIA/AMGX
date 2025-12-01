// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <global_thread_handle.h>

namespace amgx
{

class Stream
{
        cudaStream_t s;

    public:

        inline
        Stream(unsigned flags = cudaStreamNonBlocking) 
        { 
            cudaStreamCreateWithFlags(&s, flags);
            cudaCheckError();
        }

        inline
        ~Stream() 
        { 
            cudaStreamDestroy(s); 
        }

        inline
        cudaStream_t get() { return s; }
};

} // namespace amgx
