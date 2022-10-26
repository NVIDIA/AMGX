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

#include "testframework.h"
#include <iostream>
#include <string>
#include <vector>

#include "core.h"
#include "amg_solver.h"

namespace amgx
{

bool UnitTest::amgx_intialized = false;

UnitTest::UnitTest(const char *name, AMGX_Mode mode, const char *kwords) : _name(name), _failed(false), _forge_ahead(false), _last_seed(0), _failed_bit(false), _mode(mode)
{
    if (kwords != NULL)
    {
        add_keywords(std::string(kwords));
    }

    _assert_ss.clear();
    UnitTestDriverFramework::framework().register_test(this);
}

void UnitTest::assert_finite(double a, const char *filename, int lineno)
{
    if (isnan(a))
    {
        _assert_ss << "[ASSERT] " << " in test " << _name.c_str() << " at " << filename << " lineno: " << lineno << " last rand seed: " << _last_seed << std::endl;

        if (_err_stream.str().length()) { _assert_ss <<  _err_stream.str(); }

        set_failed();

        if (!_forge_ahead)
        {
            throw UnitTestFailedException();
        }
    }

    reset_err_buffer();
}

std::string UnitTest::base_keywords()
{
    return "default";
}

std::string UnitTest::custom_launch_line()
{
    return "";
}

void UnitTest::assert_true( bool a, const char *filename, int lineno)
{
    if (!a)
    {
        _assert_ss << "[ASSERT] " << " in test " << _name.c_str() << " at " << filename << " lineno: " << lineno << " last rand seed: " << _last_seed << std::endl;

        if (_err_stream.str().length()) { _assert_ss <<  _err_stream.str(); }

        set_failed();

        if (!_forge_ahead)
        {
            throw UnitTestFailedException();
        }
    }

    reset_err_buffer();
}

void UnitTest::assert_never( const char *filename, int lineno)
{
    _assert_ss << "[ASSERT] " << " in test " << _name.c_str() << " at " << filename << " lineno: " << lineno << " last rand seed: " << _last_seed << std::endl;

    if (_err_stream.str().length()) { _assert_ss <<  _err_stream.str(); }

    set_failed();

    if (!_forge_ahead)
    {
        throw UnitTestFailedException();
    }

    reset_err_buffer();
}

int UnitTest::start_test()
{
    {
        randomize( 1 );

        try
        {
            reset_err_buffer();
            start();

            for (int i = 0; i < get_configuration().repeats; ++i)
            {
                run();
                cudaDeviceSynchronize();
                cudaCheckError();
            }

            end();
            cudaDeviceSynchronize();
            cudaCheckError();
        }
        catch (UnitTestFailedException &e)
        {
            set_failed();
        }
        catch (amgx_exception &e)
        {
            _assert_ss << "[EXCEPTION] in test " << _name  << " :Caught amgx exception " << e.what() << " at " << e.where();
            set_failed();
        }
        catch (amgx::thrust::system_error &e)
        {
            _assert_ss << "[EXCEPTION] in test " << _name << ": Thrust failure: " << std::string(e.what());
            set_failed();
        }
        catch (amgx::thrust::system::detail::bad_alloc &e)
        {
            _assert_ss << "[EXCEPTION] in test " << _name << ": Not enough memory for thrust call: " << std::string(e.what());
            set_failed();
        }
        catch (std::bad_alloc &e)
        {
            _assert_ss << "[EXCEPTION] in test " << _name << ": Not enough memory: " << std::string(e.what());
            set_failed();
        }
        catch (std::exception &e)
        {
            _assert_ss << "[EXCEPTION] in test " << _name << ": Unknown exception: " << std::string(e.what());
            set_failed();
        }
        catch (...)
        {
            std::stringstream ss;
            _assert_ss << "[EXCEPTION] in test " << _name << " :Caught unhandled exception";
            set_failed();
        }
    }

    if (_failed)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

void UnitTest::randomize(int seed)
{
    if (seed == -1)
    {
        if (get_configuration().random_seed == -1 )
        {
            _last_seed = time(NULL);
        }
        else
        {
            _last_seed = get_configuration().random_seed;
        }
    }
    else
    {
        _last_seed = seed;
    }

    srand(_last_seed);
}

UnitTestConfiguration &UnitTest::get_configuration()
{
    static UnitTestConfiguration cfg;
    return cfg;
}

} // end namespace
