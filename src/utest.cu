// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "testframework.h"
#include "basic_types.h"
#include "amg_signal.h"
#include "amg_solver.h"
#include "misc.h"
#include <amgx_cusparse.h>
#include <amgx_cublas.h>


using namespace amgx;

std::string utests_stat_fname = std::string(".utestsnums");

void output_callback(const char *msg, int length)
{
}

void do_error()
{
    printf("utest [option] [test1] [test2] ...\n");
    printf("Options are:\n");
    printf(" --help             Print this message\n");
    printf(" --mode MODE        Test specified mode. Available modes: hDDI, hDFI, hFFI, dDDI, dDFI, dFFI. All by default\n");
    printf(" --all              Run all available tests. If no tests are specified, all test will be launched as well\n");
    printf(" --data PATH        Pass a custom path (absolute or relative) to the folder with external matrices. Default is '../../test_data/'\n");
    printf(" --repeat N         Repeat all tests N times\n");
    printf(" --seed X           Use number X as a forced random seed\n");
    printf(" --key A B C ...    Use only tests with keys A B C ...whatever...\n");
    printf(" --verbose          Verbose mode\n");
    printf("\n");
    printf("Available tests are:\n");
    const std::set<std::string> &names = UnitTestDriverFramework::framework().get_all_names();

    for (std::set<std::string>::const_iterator iter = names.begin(); iter != names.end(); ++iter)
    {
        std::cout << *iter << std::endl;
    }

    exit(-1);
}

int main(int argc, char **argv)
{
    int dev_cnt;
    UnitTestConfiguration cfg;
    cudaGetDeviceCount(&dev_cnt);
    int unprocessed_args = argc - 1;
    int cur_arg = 1;
    std::vector<std::string> modes;
#define AMGX_CASE_LINE(CASE) modes.push_back( std::string(ModeString<CASE>::getName()) );
    AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
    AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
    bool modes_set = false, all = false;
    std::vector<std::string> params;
    std::vector<std::string> keys;

    while (cur_arg < argc && argv[cur_arg][0] == '-')
    {
        if (strcmp(argv[cur_arg], "--repeat") == 0)
        {
            params.push_back(std::string(argv[cur_arg]));
            cur_arg++;
            unprocessed_args--;

            if (cur_arg < argc)
            {
                cfg.repeats = atoi(argv[cur_arg]);
                params.push_back(std::string(argv[cur_arg]));
            }
            else { do_error(); }
        }
        else if (strcmp(argv[cur_arg], "--seed") == 0)
        {
            params.push_back(std::string(argv[cur_arg]));
            cur_arg++;
            unprocessed_args--;

            if (cur_arg < argc)
            {
                cfg.random_seed = atoi(argv[cur_arg]);
                params.push_back(std::string(argv[cur_arg]));
            }
            else { do_error(); }
        }
        else if (strcmp(argv[cur_arg], "--log") == 0)
        {
            params.push_back(std::string(argv[cur_arg]));
            cur_arg++;
            unprocessed_args--;

            if (cur_arg < argc)
            {
                cfg.log_file = std::string(argv[cur_arg]);
                params.push_back(std::string(argv[cur_arg]));
            }
            else { do_error(); }
        }
        else if (strcmp(argv[cur_arg], "--data") == 0)
        {
            params.push_back(std::string(argv[cur_arg]));
            cur_arg++;
            unprocessed_args--;

            if (cur_arg < argc)
            {
                params.push_back(std::string(argv[cur_arg]));
                cfg.data_folder = std::string(argv[cur_arg]);

                if ( (cfg.data_folder[cfg.data_folder.length() - 1] != '/') && (cfg.data_folder[cfg.data_folder.length() - 1] != '\\') )
                {
                    cfg.data_folder += '/';
                }
            }
            else { do_error(); }
        }
        else if (strcmp(argv[cur_arg], "--key") == 0)
        {
            cur_arg++;
            unprocessed_args--;

            // untill the next CL parameter or until last parameter
            while (unprocessed_args > 0 && *(argv[cur_arg]) != '-')
            {
                keys.push_back(std::string(argv[cur_arg]));
                cur_arg++;
                unprocessed_args--;
            }

            continue;
        }
        else if (strcmp(argv[cur_arg], "--mode") == 0)
        {
            params.push_back(std::string(argv[cur_arg]));
            cur_arg++;
            unprocessed_args--;

            if (!modes_set)
            {
                modes_set = true;
                modes.clear();
            }

            params.push_back(std::string(argv[cur_arg]));

            if (strncmp(argv[cur_arg], "hDDI", 100) == 0)
            {
                modes.push_back("hDDI");
            }
            else if (strncmp(argv[cur_arg], "hDFI", 100) == 0)
            {
                modes.push_back("hDFI");
            }
            else if (strncmp(argv[cur_arg], "hFFI", 100) == 0)
            {
                modes.push_back("hFFI");
            }
            else if (strncmp(argv[cur_arg], "dDDI", 100) == 0)
            {
                modes.push_back("dDDI");
            }
            else if (strncmp(argv[cur_arg], "dDFI", 100) == 0)
            {
                modes.push_back("dDFI");
            }
            else if (strncmp(argv[cur_arg], "dFFI", 100) == 0)
            {
                modes.push_back("dFFI");
            }
            else if (strncmp(argv[cur_arg], "hZZI", 100) == 0)
            {
                modes.push_back("hZZI");
            }
            else if (strncmp(argv[cur_arg], "hZCI", 100) == 0)
            {
                modes.push_back("hZCI");
            }
            else if (strncmp(argv[cur_arg], "hCCI", 100) == 0)
            {
                modes.push_back("hCCI");
            }
            else if (strncmp(argv[cur_arg], "dZZI", 100) == 0)
            {
                modes.push_back("dZZI");
            }
            else if (strncmp(argv[cur_arg], "dZCI", 100) == 0)
            {
                modes.push_back("dZCI");
            }
            else if (strncmp(argv[cur_arg], "dCCI", 100) == 0)
            {
                modes.push_back("dCCI");
            }
            else
            {
                printf("Incorrect mode: %s\n", argv[cur_arg]);
                exit(1);
            }
        }
        else if (strcmp(argv[cur_arg], "--help") == 0)
        {
            do_error();
        }
        else if (strcmp(argv[cur_arg], "--all") == 0)
        {
            params.push_back(std::string(argv[cur_arg]));
            all = true;
        }
        else if (strcmp(argv[cur_arg], "--verbose") == 0)
        {
            params.push_back(std::string(argv[cur_arg]));
            cfg.verbose = true;
        }
        else if (strcmp(argv[cur_arg], "--child") == 0)
        {
            cfg.is_child = true;
        }
        else if (strcmp(argv[cur_arg], "--suppress_color") == 0)
        {
            params.push_back(std::string(argv[cur_arg]));
            cfg.suppress_color = true;
        }

        cur_arg++;
        unprocessed_args--;
    }

    cfg.program_name = std::string(argv[0]);

// setup application
    if (cfg.is_child && !cfg.verbose)
    {
        amgx::amgx_output = &output_callback;
    }

    UnitTestDriverFramework::framework().set_configuration(cfg);
    SignalHandler::hook();
    int total = 0, failed = 0;

    if (cfg.is_child)
    {
        // call this once to force everything to initialize so any timing results are not skewed
        cudaSetDevice(0);
        cudaCheckError();
        cudaFree(0);
        cudaCheckError();
        Cusparse &c = Cusparse::get_instance();
        Cublas::get_handle();
        UnitTestDriverFramework::framework().do_work();
        c.destroy_handle();
        Cublas::destroy_handle();
        cudaDeviceReset();
        return 0;
    }
    else
    {
        int res = 0;

        if (all || unprocessed_args == 0)
        {
            res = UnitTestDriverFramework::framework().run_all_tests(total, failed, modes, keys, params);
        }
        else
        {
            std::vector<std::string> tests;

            for (int i = argc - unprocessed_args; i < argc; i++)
            {
                tests.push_back(argv[i]);
            }

            res = UnitTestDriverFramework::framework().run_tests(total, failed, modes, keys, tests, params);
        }

        if (res != 0)
        {
            // some itnernal error.
        }
        else
        {
            // output file with statistics
            std::ofstream stats(utests_stat_fname.c_str());
            stats << total << " " << failed;
            stats.close();
        }
    }

    cudaDeviceReset();
    return failed;
}
