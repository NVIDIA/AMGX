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
namespace template_plugin
{
template <class Matrix> class TemplateCycle;
}
}

#include <cycles/fixed_cycle.h>

namespace amgx
{
namespace template_plugin
{

template <class Matrix>
class TemplateCycle : public FixedCycle<Matrix>
{
    public:
        typedef Vector<typename Matrix::value_type, typename Matrix::memory_space> VVector;
        TemplateCycle(AMG<Matrix> *amg, AMG_Level<Matrix> *level, const VVector &b, VVector &x)
        {
            cycle(amg, level, b, x);
        }
        void nextCycles(AMG<Matrix> *amg, AMG_Level<Matrix> *level, const VVector &b, VVector &x);
};

template<class Matrix>
class TemplateCycleFactory : public CycleFactory<Matrix>
{
    public:
        typedef Vector<typename Matrix::value_type, typename Matrix::memory_space> VVector;
        Cycle<Matrix> *create(AMG<Matrix> *amg, AMG_Level<Matrix> *level, const VVector &b, VVector &x) { return new TemplateCycle<Matrix>(amg, level, b, x); }
};
} // namespace template_plugin
} // namespace amgx
