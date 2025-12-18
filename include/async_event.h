// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace amgx
{

class AsyncEvent
{
    public:
        AsyncEvent() : async_event(NULL) { }
        AsyncEvent(int size) : async_event(NULL) { cudaEventCreate(&async_event); }
        ~AsyncEvent() { if (async_event != NULL) cudaEventDestroy(async_event); }

        void create() { cudaEventCreate(&async_event); }
        void record(cudaStream_t s = 0)
        {
            if (async_event == NULL)
            {
                cudaEventCreate(&async_event);    // check if we haven't created the event yet
            }

            cudaEventRecord(async_event, s);
        }
        void sync()
        {
            cudaEventSynchronize(async_event);
        }
    private:
        cudaEvent_t async_event;
};

}
