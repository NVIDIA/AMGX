// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cassert>
#include "thread_manager.h"
#include "vector.h"

namespace amgx
{

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////

ThreadWorker::ThreadWorker(ThreadManager *manager, int skills) :
    m_manager(manager),
    m_skills(skills)
{
}

ThreadWorker::~ThreadWorker()
{
}

float ThreadWorker::estimate_workload()
{
    return 0.0;
}

void ThreadWorker::push_task(AsyncTask *task)
{
}

void ThreadWorker::wait_empty()
{
}

void ThreadWorker::run()
{
}

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////

void ThreadManager::setup_streams( int num_streams, bool priority, bool serialize )
{
}

void ThreadManager::join_threads()
{
}

void ThreadManager::wait_threads()
{
}

void ThreadManager::spawn_threads(size_t pool_size,
                                  size_t max_alloc_size)
{
}

void ThreadManager::push_work(AsyncTask *task, bool use_cnp)
{
}

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////

void InitTask::exec()
{
}

}   // namespace amgx

