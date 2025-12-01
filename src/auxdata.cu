// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <auxdata.h>

#ifndef _WIN32
#include <cxxabi.h>
const std::string type_demangle(const char *name)
{
    int status = -4;
    char *res = abi::__cxa_demangle(name, NULL, NULL, &status);
    const char *const demangled_name = (status == 0) ? res : name;
    std::string ret_val(demangled_name);
    free(res);
    return ret_val;
}
#else
const std::string type_demangle(const char *name)
{
    return std::string(name);
}
#endif

namespace amgx
{

void AuxDB::printExistingData() const
{
    printf("Storing parameters with names:\n");
    ParamPtrDB::const_iterator iter;

    for (iter = ptrparams.begin(); iter != ptrparams.end(); ++iter)
    {
        printf("%s\n", iter->first.c_str());
    }

    ParamDB::const_iterator iter2;

    for (iter2 = params.begin(); iter2 != params.end(); ++iter2)
    {
        printf("%s\n", iter2->first.c_str());
    }
}

bool AuxDB::hasParameter(const std::string &name) const
{
    return (params.find(name) != params.end() || ptrparams.find(name) != ptrparams.end());
}

void AuxDB::copyParameters(const AuxDB *src)
{
    // clear existing pointers?
    clearPtrs();
    params = src->params;
    ParamPtrDB::const_iterator iter;

    for (iter = src->ptrparams.begin(); iter != src->ptrparams.end(); ++iter)
    {
        AuxPtr<int> *item = new AuxPtr<int>((int *)(iter->second->Get()), false); // specify some type instead of void to avoid "delete void*" warning. This object won't delete this pointed object.
        item->force_typename(iter->second->type_name);
        ptrparams[iter->first] = item;
    }
}

void AuxDB::clearPtrs()
{
    ParamPtrDB::const_iterator iter;

    for (iter = ptrparams.begin(); iter != ptrparams.end(); ++iter)
    {
        delete iter->second;
    }

    ptrparams.clear();
}

AuxDB::~AuxDB()
{
    clearPtrs();
}

} // namespace amgx
