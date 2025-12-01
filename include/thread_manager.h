// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cuda_runtime.h>
#include <list>
#include <vector>

#include "basic_types.h"
#include "global_thread_handle.h"

namespace amgx
{

// Barebone for spawning async-like tasks
class ThreadWorker;
class ThreadManager;

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////

class AsyncTask
{
        // We want the worker to be able to set the 'm_worker' member data.
        friend class ThreadWorker;

    public:
        // Tags to indicate if a task requires some special capacity from the worker.
        enum
        {
            SKILL_NONE = 0x0, // Can be executed by all workers.
            SKILL_MPI  = 0x1  // Can only be executed by the MPI worker.
        };

    private:
        // Possible statuses of a task.
        enum
        {
            TASK_IS_READY_TO_BE_EXECUTED     = 0x1, // The task is ready to be executed,
            TASK_IS_EXECUTING                = 0x2, // The task is executing,
            TASK_IS_FINISHED                 = 0x4  // The task is finished.
        };

    private:
        // The status of the task.
        int m_status;
        // A task may have an ID.
        int m_id;
        // A task may need some special skills.
        int m_needed_skills;
        // A task is processed by a worker.
        ThreadWorker *m_worker;

    public:
        // Constructor.
        AsyncTask(int id = -1, int needed_skills = SKILL_NONE) :
            m_status(TASK_IS_READY_TO_BE_EXECUTED),
            m_id(id),
            m_needed_skills(needed_skills)
        {}

        // The terminate function indicates if it's the last task.
        virtual bool terminate() { return false; }
        // Run the task.
        virtual void exec() = 0;

        // The id of the task.
        inline int get_id() const { return m_id; }
        // The skills needed by that task.
        inline int get_needed_skills() const { return m_needed_skills; }

        // Is a task executing?
        inline bool is_executing() const { return m_status == TASK_IS_EXECUTING; }
        // Is a task finished?
        inline bool is_finished() const { return m_status == TASK_IS_FINISHED; }

    private:
        // Set the worker.
        inline void set_worker(ThreadWorker *worker) { m_worker = worker; }
};

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////

class ThreadWorker
{
        // We want to make sure only the thread manager can create workers.
        friend class ThreadManager;

        // The associated thread manager.
        ThreadManager *m_manager;

        // The thread running that worker.
        _thread_id *m_thread;
        // The mutex to lock the list of tasks.
        std::mutex m_mutex;

        // The skills of the thread worker.
        int m_skills;

        ThreadWorker(ThreadManager *manager, int skills = AsyncTask::SKILL_NONE);

    public:
        // Destroy a worker.
        ~ThreadWorker();

        // The associated thread manager.
        inline ThreadManager *get_manager() { return m_manager; }

        // The skills of that worker.
        inline int get_skills() const { return m_skills; }

        // A worker determines its load.
        float estimate_workload();
        // Push a task in the task queue.
        void push_task(AsyncTask *task);
        // Run the tasks until it finds a termination task.
        void run();
        // Waits until queue is empty
        void wait_empty();
};

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////

class ThreadManager
{
        // The CUDA streams. The latest stream is the high priority stream (if any).
        std::vector<cudaStream_t> m_cuda_streams;
        // Do we run the task sequentially?
        bool m_serialize_mode;
        // The work queues.
        std::vector<ThreadWorker *> m_workers;

    public:
        // Create the thread manager. The streams are not created (???).
        ThreadManager() : m_serialize_mode(false) {}

        // Push a new task. The manager tries to find the "best" task queue.
        void push_work(AsyncTask *func, bool use_cnp = false);

        // Create CUDA streams and work threads.
        void setup_streams(int num_streams = 0, bool priority = false, bool serialize = false);

        // Wait until all threads complete their queues
        void wait_threads();

        // Spawn and join threads. 
        void spawn_threads(size_t pool_size,
                           size_t max_alloc_size);
        void join_threads();
};

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////

class InitTask : public AsyncTask
{
        // The device.
        int m_device;
        // The CUDA stream.
        cudaStream_t m_stream;
        // A mutex to protect thrust global handles.
        int *m_mutex;

        // device pool size for EACH threads
        size_t m_pool_size;
        // device pool max alloc size for a  single allocation
        size_t m_max_alloc_size;

        size_t m_max_size;

    public:
        InitTask(int device,
                 cudaStream_t stream,
                 int *mutex,
                 size_t pool_size,
                 size_t max_alloc_size) :
            m_device(device),
            m_stream(stream),
            m_mutex(mutex),
            m_pool_size(pool_size),
            m_max_alloc_size(max_alloc_size),
            m_max_size(0)
        {}

        void exec();
};

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////

class TerminationTask : public AsyncTask
{
    public:
        bool terminate() { return true; }
        void exec() {}
};

} // namespace amgx

