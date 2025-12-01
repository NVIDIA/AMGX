// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace amgx
{
class MemoryInfo
{
    public:
        static float getTotalMemory()
        {
            size_t free;
            size_t total;
            cudaMemGetInfo(&free, &total);
            return total / 1024.0 / 1024 / 1024;
        }

        static size_t getFreeMemory()
        {
            size_t free;
            size_t total;
            cudaMemGetInfo(&free, &total);
            return free / 1024.0 / 1024 / 1024;
        }

        static float getMaxMemoryUsage()
        {
            return max_allocated / 1024.0 / 1024 / 1024;
        }

        static void updateMaxMemoryUsage()
        {
            size_t free;
            size_t total;
            cudaMemGetInfo(&free, &total);
            size_t allocated = total - free;

            if (allocated > max_allocated)
            {
                max_allocated = allocated;
            }
        }
    private:
        static size_t max_allocated;
};
}
