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

#include <amgx_timer.h>

#include <sstream>

#ifdef AMGX_WITH_MPI
#include <mpi.h>
#endif

#ifdef AMGX_USE_VAMPIR_TRACE
#include <vt_user.h>
#endif

namespace amgx
{

#ifdef AMGX_USE_CPU_PROFILER

std::ostream &Profiler_entry::print(std::ostream &out, int depth, int max_depth, double total_time, double parent_time) const
{
    for ( int i = 0 ; i < depth ; ++i )
    {
        out << "| ";
    }

    out << std::setw(WIDTH) << std::setfill('.') << std::left << m_name;

    for ( int i = 0 ; i < max_depth - depth ; ++i )
    {
        out << "..";
    }

    out << " |" << std::setw(10) << std::setfill(' ') << std::right << std::fixed << std::setprecision(3) << get_time_in_ms()  << " |";
    double abs_prct = 100.0 * get_time_in_ms() / total_time;
    out << std::setw(7) << std::right << std::fixed << std::setprecision(2) << abs_prct << " % |";
    double rel_prct = 100.0 * get_time_in_ms() / parent_time;
    out << std::setw(7) << std::right << std::fixed << std::setprecision(2) << rel_prct << " % |" << std::endl;

    // Early exit if no child.
    if ( m_children.empty() )
    {
        return out;
    }

    // Children.
    double sum(0.0);

    for ( int i = 0, n_children = m_children.size() ; i < n_children ; ++i )
    {
        m_children[i].print(out, depth + 1, max_depth, total_time, get_time_in_ms());
        sum += m_children[i].get_time_in_ms();
    }

    // Unknown fraction of time.
    for ( int i = 0 ; i < depth + 1 ; ++i )
    {
        out << "| ";
    }

    out << std::setw(WIDTH) << std::setfill('.') << std::left << "self (excluding children)";

    for ( int i = 0 ; i < max_depth - depth - 1 ; ++i )
    {
        out << "..";
    }

    double unknown_time = get_time_in_ms() - sum;
    out << " |" << std::setw(10) << std::setfill(' ') << std::right << std::fixed << std::setprecision(3) << unknown_time << " |";
    abs_prct = 100.0 * unknown_time / total_time;
    out << std::setw(7) << std::right << std::fixed << std::setprecision(2) << abs_prct << " % |";
    rel_prct = 100.0 * unknown_time / get_time_in_ms();
    out << std::setw(7) << std::right << std::fixed << std::setprecision(2) << rel_prct << " % |";

    if ( rel_prct >= 0.10 ) // Add a marker if >10% is unknown.
    {
        out << " *!!*";
    }

    out << std::endl;
    return out;
}

Profiler_tree &Profiler_tree::get_instance()
{
    static Profiler_tree s_instance;
    return s_instance;
}

Profiler_tree::Profiler_tree() : m_max_depth(0), m_root("amgx")
#ifdef AMGX_WITH_MPI
    , m_rank(-1) // We can't guarantee that MPI will be Initialized when it is built or destroyed.
#endif
{
    m_stack.reserve(32);
    m_stack.push_back(&m_root);
    m_root.start();
    m_first_start = high_resolution_clock::now();
}

Profiler_tree::~Profiler_tree()
{
    m_root.set_time((std::chrono::duration_cast<std::chrono::nanoseconds>(m_last_stop - m_first_start)).count());
    std::ostringstream buffer;
#ifdef AMGX_WITH_MPI

    if (m_rank != -1)
    {
        buffer << "amgx_cpu_profile." << m_rank << ".txt";
    }
    else
#endif
        buffer << "amgx_cpu_profile.txt";

    std::ofstream file( buffer.str().c_str(), std::ios::out );

    for ( int i = 0, end = 64 + 2 * m_max_depth ; i < end ; ++i )
    {
        file << " ";
    }

    file << " | Time (ms) | Absolute | Relative |" << std::endl;
    m_root.print(file, 0, m_max_depth, m_root.get_time_in_ms(), m_root.get_time_in_ms() );
    file.close();
}

void Profiler_tree::push( const char *name )
{
#ifdef AMGX_WITH_MPI

    if (m_rank == -1)
    {
        int flag = 0;
        MPI_Initialized(&flag); // We want to make sure MPI_Init has been called.

        if (flag)
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
        }
    }

#endif
    Profiler_entry *top_of_stack = m_stack.back();
    Profiler_entry *node = top_of_stack->add_child(name);
    m_stack.push_back(node);
    m_max_depth = std::max( m_max_depth, (int) m_stack.size() );
    node->start();
}

void Profiler_tree::pop()
{
    assert(!m_stack.empty());
    Profiler_entry *top_of_stack = m_stack.back();
    top_of_stack->stop();
    m_stack.pop_back();
    m_last_stop = high_resolution_clock::now();
}

void Profiler_tree::mark(const char *c_name, const char *msg)
{
#ifdef AMGX_USE_VAMPIR_TRACE
    typedef std::map<std::string, int>::iterator Iterator;
    std::string name(c_name);
    Iterator it = m_markers.find(name);
    int tag = 0;

    if ( it == m_markers.end() )
    {
        tag = VT_User_marker_def__(c_name, VT_MARKER_TYPE_HINT);
        m_markers[name] = tag;
    }
    else
    {
        tag = it->second;
    }

    VT_User_marker__(tag, msg);
#endif
}

Profiler_raii::Profiler_raii( const char *name, const char *filename, int lineno ) : m_name(name)
{
    Profiler_tree &tree = Profiler_tree::get_instance();
    tree.push(name);
#ifdef AMGX_USE_VAMPIR_TRACE
    VT_User_start__(m_name.c_str(), filename, lineno);
#endif
}

Profiler_raii::~Profiler_raii()
{
#ifdef AMGX_USE_VAMPIR_TRACE
    VT_User_end__(m_name.c_str());
#endif
    Profiler_tree &tree = Profiler_tree::get_instance();
    tree.pop();
}

#endif // defined AMGX_USE_CPU_PROFILER


static TimerMap amgxTimers;
TimerMap &getTimers()
{
    return amgxTimers;
}

} // end namespace amgx


