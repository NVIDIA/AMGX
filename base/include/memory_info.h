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
