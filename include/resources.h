// SPDX-FileCopyrightText: 2013 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace amgx
{
class Resources;
}

#include "amg_config.h"
#include "thread_manager.h"
#ifdef AMGX_WITH_MPI
#include "mpi.h"
#endif

namespace amgx
{

class Resources
{
        ThreadManager *m_tmng;
        std::vector<int> m_devices;
        AMG_Config *m_cfg;
        bool m_cfg_self;
        size_t m_pool_size;
        size_t m_pool_size_limit;
        size_t m_max_alloc_size;
        size_t m_scaling_factor;
        size_t m_scaling_threshold;
        size_t m_root_pool_size;
        size_t m_root_pool_expanded;
        bool m_handle_errors;
        int m_verbosity_level;

        int m_num_streams;
        int m_high_priority_stream;
        int m_serialize_threads;
#ifdef AMGX_WITH_MPI
        MPI_Comm *m_mpi_comm;
#endif
    public:
        Resources(AMG_Configuration *cfg, void *comm, int device_num, const int *devices);
        Resources();    // simplified constructor for single GPU, uses device 0, default config options (no async-stuff, etc.)
        ~Resources();

        ThreadManager *get_tmng() { return m_tmng; }
        AMG_Config *getResourcesConfig() { return m_cfg; }
#ifdef AMGX_WITH_MPI
        MPI_Comm *getMpiComm() { return m_mpi_comm; }
#endif
        int getDevice(int device_num) const { return m_devices[device_num]; }
        bool getHandleErrors() const { return m_handle_errors; }
        size_t getPoolSize() const { return m_pool_size; }
        void expandRootPool();

        void warning(const std::string s) const;
};

void free_resources(); //failsafe inteface to releasing internal static memory pools in case of early exit

};
