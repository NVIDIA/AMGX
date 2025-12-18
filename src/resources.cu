// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <resources.h>
#include <core.h>

#include <amgx_cusparse.h>
#include <amgx_cublas.h>

namespace amgx
{

void allocate_resources(size_t pool_size,
                        size_t max_alloc_size,
                        size_t scaling_factor,
                        size_t scaling_threshold,
                        size_t max_size)
{
    // create main memory pools (GPU specific resources)
    // its safe to do here becasue cudaSetDevice is invoked right before communicator class is created
    if ( !memory::hasPinnedMemoryPool() )
    {
        memory::setPinnedMemoryPool( new memory::PinnedMemoryPool() );
    }

    if ( !memory::hasDeviceMemoryPool() )
        memory::setDeviceMemoryPool( new memory::DeviceMemoryPool(pool_size,
                                     max_alloc_size,
                                     max_size) );

    memory::setMallocScalingFactor(scaling_factor);
    memory::setMallocScalingThreshold(scaling_threshold);
}

void free_resources()
{
    memory::destroyAllPinnedMemoryPools();
    memory::destroyAllDeviceMemoryPools();
}

void Resources::warning(const std::string s) const
{
    if (m_verbosity_level == 3)
    {
        std::cout << "WARNING: " << s << std::endl;
    }
}

// simplified resources constructor - single GPU only
Resources::Resources() : m_cfg_self(true), m_root_pool_expanded(false), m_tmng(nullptr)
{
    m_cfg = new AMG_Config;
    m_devices.clear();
    m_devices.push_back(0);
    cudaSetDevice(0);
    cudaCheckError();
    cudaFree(0);
    cudaCheckError();
    std::string solver_value, solver_scope, default_scope;
    m_cfg->getParameter<std::string>("solver", solver_value, "default", solver_scope);
    m_cfg->getParameter<size_t>("device_mem_pool_size", m_pool_size, "default", solver_scope);
    m_cfg->getParameter<size_t>("device_mem_pool_size_limit", m_pool_size_limit, "default", solver_scope);
    m_cfg->getParameter<size_t>("device_consolidation_pool_size", m_root_pool_size, "default", solver_scope);
    m_cfg->getParameter<size_t>("device_mem_pool_max_alloc_size", m_max_alloc_size, "default", solver_scope);
    m_cfg->getParameter<size_t>("device_alloc_scaling_factor", m_scaling_factor, "default", solver_scope);
    m_cfg->getParameter<size_t>("device_alloc_scaling_threshold", m_scaling_threshold, "default", solver_scope);
    m_cfg->getParameter<int>("verbosity_level", m_verbosity_level, "default", solver_scope);
    m_cfg->getParameter<int>("num_streams", m_num_streams, "default", solver_scope);
    m_cfg->getParameter<int>("high_priority_stream", m_high_priority_stream, "default", solver_scope);
    m_cfg->getParameter<int>("serialize_threads", m_serialize_threads, "default", solver_scope);
    amgx::allocate_resources(m_pool_size, m_max_alloc_size, m_scaling_factor, m_scaling_threshold, m_pool_size_limit);
    // setup NV libraries
    Cusparse &c = Cusparse::get_instance();
    c.set_determinism_flag(m_cfg->getParameter<int>("determinism_flag", "default") != 0);
    Cublas::get_handle();
    // create and initialize thread manager
    //m_tmng = new ThreadManager();
    //m_tmng->setup_streams();
    // spawn threads
    //m_tmng->spawn_threads(m_pool_size, m_max_alloc_size);
    // reset settings to normal
    memory::setAsyncFreeFlag(false);
    memory::setDeviceMemoryPoolFlag(true);
    m_handle_errors = m_cfg->getParameter<int>("exception_handling", default_scope);
}

Resources::Resources(AMG_Configuration *cfg, void *comm, int device_num, const int *devices) : 
    m_handle_errors(true), m_cfg_self(false), m_root_pool_expanded(false), m_tmng(nullptr)
{
    m_devices.clear();
    m_cfg = cfg->getConfigObject();
    std::string solver_value, solver_scope;
    m_cfg->getParameter<std::string>("solver", solver_value, "default", solver_scope);
    m_cfg->getParameter<size_t>("device_mem_pool_size", m_pool_size, "default", solver_scope);
    m_cfg->getParameter<size_t>("device_mem_pool_size_limit", m_pool_size_limit, "default", solver_scope);
    m_cfg->getParameter<size_t>("device_consolidation_pool_size", m_root_pool_size, "default", solver_scope);
    m_cfg->getParameter<size_t>("device_mem_pool_max_alloc_size", m_max_alloc_size, "default", solver_scope);
    m_cfg->getParameter<size_t>("device_alloc_scaling_factor", m_scaling_factor, "default", solver_scope);
    m_cfg->getParameter<size_t>("device_alloc_scaling_threshold", m_scaling_threshold, "default", solver_scope);
    m_cfg->getParameter<int>("verbosity_level", m_verbosity_level, "default", solver_scope);
    m_cfg->getParameter<int>("num_streams", m_num_streams, "default", solver_scope);
    m_cfg->getParameter<int>("high_priority_stream", m_high_priority_stream, "default", solver_scope);
    m_cfg->getParameter<int>("serialize_threads", m_serialize_threads, "default", solver_scope);

    // loop over all devices
    for (int i = 0; i < device_num; i++)
    {
        // add device to our list
        m_devices.push_back(devices[i]);
        // select current device
        cudaSetDevice(devices[i]);
        cudaCheckError();
        // create context
        cudaFree(0);
        cudaCheckError();
        // allocate resources
        amgx::allocate_resources(m_pool_size, m_max_alloc_size, m_scaling_factor, m_scaling_threshold, m_pool_size_limit);
        m_handle_errors = m_cfg->getParameter<int>("exception_handling", solver_scope);
    }

    // setup NV libraries
    Cusparse &c = Cusparse::get_instance();
    c.set_determinism_flag(m_cfg->getParameter<int>("determinism_flag", "default") != 0);
    Cublas::get_handle();
    // create communicator
    // create and initialized thread manager
    //m_tmng = new ThreadManager();
    //m_tmng->setup_streams(m_num_streams, m_high_priority_stream, m_serialize_threads);
    // spawn threads
    //m_tmng->spawn_threads(m_pool_size, m_max_alloc_size);
    // reset settings to normal
    memory::setAsyncFreeFlag(false);
    memory::setDeviceMemoryPoolFlag(true);
    // create communicator
#ifdef AMGX_WITH_MPI
    m_mpi_comm = (MPI_Comm *) comm;
#endif
}

Resources::~Resources()
{
    // select device 0
    cudaSetDevice(m_devices[0]);
    cudaCheckError();
    // terminate threads
    // m_tmng->join_threads();
    // delete m_tmng;
    // destroy NV libraries
    Cusparse &c = Cusparse::get_instance();
    c.destroy_handle();
    Cublas::destroy_handle();

    // loop over all devices
    for (int i = 0; i < m_devices.size(); i++)
    {
        // select current device
        cudaSetDevice(m_devices[i]);
        cudaCheckError();
        // free resources
        amgx::free_resources();
    }

    if (m_cfg_self)
    {
        delete m_cfg;
    }
}

void Resources::expandRootPool()
{
    if (!m_root_pool_expanded)
    {
        for (int i = 0; i < m_devices.size(); i++)
        {
            // select current device
            cudaSetDevice(m_devices[i]);
            cudaCheckError();
            memory::expandDeviceMemoryPool(m_root_pool_size, m_max_alloc_size);
        }

        m_pool_size += m_root_pool_size;
        m_root_pool_expanded = true;
        // resetup thread manager
        m_tmng->join_threads();
        delete m_tmng;
        m_tmng = new ThreadManager();
        m_tmng->setup_streams(m_num_streams, m_high_priority_stream, m_serialize_threads);
        m_tmng->spawn_threads(m_pool_size, m_max_alloc_size);
    }
}

}

