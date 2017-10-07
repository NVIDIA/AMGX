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

#ifdef _WIN32
#include <windows.h>
#include <process.h>
const char *empty = "";
const char *_COLOR_GREEN (bool suppress_color)
{
    if (!suppress_color)
    {
        HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
        SetConsoleTextAttribute(hStdout, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
    }

    return empty;
}
const char *_COLOR_RED (bool suppress_color)
{
    if (!suppress_color)
    {
        HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
        SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_INTENSITY);
    }

    return empty;
}
const char *_COLOR_NONE (bool suppress_color)
{
    if (!suppress_color)
    {
        HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
        SetConsoleTextAttribute(hStdout, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
    }

    return empty;
}
#else
const char *red_color = "\033[31m";
const char *green_color = "\033[32m";
const char *no_color = "\033[0m";
const char *empty = "";
const char *_COLOR_GREEN (bool suppress_color)
{
    return suppress_color ? empty : green_color;
}
const char *_COLOR_RED (bool suppress_color)
{
    return suppress_color ? empty : red_color;
}
const char *_COLOR_NONE (bool suppress_color)
{
    return suppress_color ? empty : no_color;
}
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#endif

#define COLOR_RED _COLOR_RED(UnitTest::get_configuration().suppress_color)
#define COLOR_NONE _COLOR_NONE(UnitTest::get_configuration().suppress_color)
#define COLOR_GREEN _COLOR_GREEN(UnitTest::get_configuration().suppress_color)


namespace amgx
{

void UnitTestDriver::register_test(UnitTest *test)
{
    std::map< std::string, std::map<std::string, UnitTest *> >::iterator test_by_name = tests.find(test->name());

    if (test_by_name == tests.end())
    {
        tests[test->name()] = std::map<std::string, UnitTest *>();
        test_by_name = tests.find(test->name());
    }

    switch (test->getMode())
    {
#define AMGX_CASE_LINE(CASE) case CASE: test_by_name->second[std::string(ModeString<CASE>::getName())] = test; break;
            AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
    }
}

int UnitTestDriver::run_test (const std::string &name, const std::string &mode, std::ostream &logfile)
{
    logfile << name << " " << mode << ": ";
    logfile.flush();
    std::map< std::string, std::map<std::string, UnitTest *> >::iterator test_by_name = tests.find(name);

    if (test_by_name == tests.end() || test_by_name->second.find(mode) == test_by_name->second.end())
    {
        logfile << "[ERROR] Haven't found this test in the amgx" << std::endl;
        logfile.flush();
        return -1;
    }
    else
    {
        int res = 0;
        UnitTest *test = test_by_name->second[mode];
        test->start_test();

        if (test->failed())
        {
            test->print_assert(logfile);
            logfile << std::endl;
            res = 1;
        }
        else
        {
            logfile << "[PASSED]" << std::endl;
        }

        test_by_name->second.erase(mode);
        logfile.flush();
        return res;
    }
}

int UnitTestDriver::run_tests (const std::vector< std::pair<std::string, std::string> > &tests, std::ostream &logfile)
{
    std::ofstream process_log(f_worker_processed.c_str(), std::ios_base::app);
    amgx::initialize();
    amgx::initializePlugins();
    UnitTest::amgx_intialized = true;
    bool first_try = true;

    for (unsigned int i = 0; i < tests.size(); i++)
    {
        if (!UnitTest::amgx_intialized)
        {
            amgx::initialize();
            amgx::initializePlugins();
        }

        printf("[%4d/%4d] %s\t %s : ", i + 1, (int)(tests.size()), tests[i].first.c_str(), tests[i].second.c_str());
        fflush(stdout);

        if (tests[i] != this->last_launched)
        {
            // first try of the test
            first_try = true;
            std::ofstream second_try_file(f_secondtry.c_str(), std::ios_base::out);
            second_try_file << tests[i].first.c_str() << std::endl << tests[i].second.c_str();
            second_try_file.flush();
            second_try_file.close();
        }
        else
        {
            // second try of the test
            first_try = false;
            process_log << tests[i].first.c_str() << std::endl << tests[i].second.c_str();
            process_log.flush();
            remove(f_secondtry.c_str());
            std::ofstream second_try_file_name(f_secondtry_name.c_str(), std::ios_base::out);
            second_try_file_name << tests[i].first.c_str() << std::endl << tests[i].second.c_str();
            second_try_file_name.close();
        }

        fflush(stdout);
        int res = run_test(tests[i].first, tests[i].second, logfile);

        if (res != 0)
        {
            std::cout << "[" << COLOR_RED << "FAILED" << COLOR_NONE << "]" << std::endl;

            if (first_try)
            {
                // if first try - exit and launch same test in new process
                exit(512);
            }
            else
            {
                // if second try - failed. write result to the log
                process_log << std::endl << 1 << std::endl;
            }
        }
        else
        {
            if (first_try)
            {
                process_log << tests[i].first.c_str() << std::endl << tests[i].second.c_str();
            }

            process_log << std::endl << 0 << std::endl;
            std::cout << "[" << COLOR_GREEN << "PASSED" << COLOR_NONE << "]" << std::endl;
        }

        remove(f_secondtry.c_str());
        logfile.flush();
        std::cout.flush();
    }

    if (UnitTest::amgx_intialized)
    {
        amgx::finalizePlugins();
        amgx::finalize();
    }

    UnitTest::amgx_intialized = false;
    process_log.close();
    return 0;
}

UnitTest *UnitTestDriver::get_test (const std::string &name, const std::string &mode) const
{
    std::map< std::string, std::map<std::string, UnitTest *> >::const_iterator test_by_name = tests.find(name);

    if (test_by_name == tests.end())
    {
        return NULL;
    }

    std::map<std::string, UnitTest *>::const_iterator titer = test_by_name->second.find(mode);
    return (titer == test_by_name->second.end()) ? NULL : titer->second;
}

void UnitTestDriver::print_all_tests(std::ostream &str)
{
    std::map< std::string, std::map<std::string, UnitTest *> >::iterator test_by_name = tests.begin();

    for (; test_by_name != tests.end(); ++test_by_name)
    {
        str << test_by_name->first << ": ";
        std::map<std::string, UnitTest *>::iterator test_by_mode = test_by_name->second.begin();

        for (; test_by_mode != test_by_name->second.end(); ++test_by_mode)
        {
            str << test_by_mode->first;
        }

        str << "\n";
    }
}


///////////////////////////////////////////////////////////////////////
///////                  UnitTestDriverFramework            ///////////
///////////////////////////////////////////////////////////////////////

// this thing runs workers, manages output and stuff...

UnitTestDriverFramework::~UnitTestDriverFramework()
{
}

void UnitTestDriverFramework::register_test(UnitTest *test)
{
    test_driver.register_test(test);
    all_test_names.insert(test->name());
}

void UnitTestDriverFramework::do_work()
{
    std::ifstream schedule(f_schedule.c_str());
    std::ifstream processed(f_worker_processed.c_str());
    std::vector< std::pair<std::string, std::string> > processed_tests;
    std::vector< std::pair<std::string, std::string> > to_process;

    if (processed.is_open())
    {
        while (!processed.eof())
        {
            std::string name, mode, stub;
            int tres; // we don't need processed tests result now
            std::getline(processed, name);
            std::getline(processed, mode);
            processed >> tres;
            std::getline(processed, stub);
            processed_tests.push_back(std::make_pair(name, mode));
        }

        processed.close();
    }

    std::ifstream last_launched_file(f_secondtry.c_str(), std::ios_base::in);

    if (last_launched_file.is_open())
    {
        test_driver.check_last_launch = true;
        std::string name, mode;
        std::getline(last_launched_file, name);
        std::getline(last_launched_file, mode);
        test_driver.last_launched = std::make_pair(name, mode);
        printf("Launching test %s %s in the new process for the second time\n", name.c_str(), mode.c_str());
        last_launched_file.close();
        remove(f_secondtry.c_str());
    }

    if (schedule.is_open())
    {
        while (!schedule.eof())
        {
            std::string name, mode;
            std::getline(schedule, name);
            std::getline(schedule, mode);

            if (std::find(processed_tests.begin(), processed_tests.end(), std::make_pair(name, mode)) == processed_tests.end())
            {
                to_process.push_back(std::make_pair(name, mode));
            }
        }

        schedule.close();
    }

    std::ofstream unit_test_log(f_assert_log.c_str(), std::ios_base::app);
    test_driver.run_tests(to_process, unit_test_log);
    unit_test_log.close();
    printf("Done!\n");
    fflush(stdout);
    // retrieve all required tests
}

int UnitTestDriverFramework::run_tests(int &total, int &failed, const std::vector<std::string> &str_modes, const std::vector<std::string> &kwords, const std::vector<std::string> &tests, const std::vector<std::string> &params)
{
    // clear stuff
    remove(f_schedule.c_str());
    remove(f_assert_log.c_str());
    remove(f_worker_processed.c_str());
    remove(f_statfile.c_str());
    remove(f_secondtry.c_str());
    remove(f_secondtry_name.c_str());
    // write schedule
    total = 0;
    std::ofstream schedule(f_schedule.c_str(), std::ios_base::out | std::ios_base::trunc);
    bool need_cr = false;

    for (unsigned int i = 0; i < tests.size(); i++)
    {
        for (unsigned int j = 0; j < str_modes.size(); j++)
        {
            UnitTest *test = test_driver.get_test(tests[i], str_modes[j]);

            if (test != NULL)
            {
                bool write_test = true;

                if (kwords.size() > 0)
                {
                    write_test = false;
                    std::vector<std::string> str_keywords;
                    split(test->getKeywords(), ';', str_keywords);

                    for (unsigned int k = 0; k < str_keywords.size(); k++)
                    {
                        if (std::find(kwords.begin(), kwords.end(), str_keywords[k]) != kwords.end())
                        {
                            write_test = true;
                            break;
                        }
                    }
                }

                if (write_test)
                {
                    if (need_cr)
                    {
                        schedule << std::endl;
                    }

                    schedule << tests[i] << std::endl << str_modes[j];
                    total ++;
                    need_cr = true;
                }
            }
        }
    }

    schedule.close();
    //printf("Going to process total %d tests\n", total);
    int processed = 0;
    const UnitTestConfiguration &cfg = UnitTest::get_configuration();
    // creating arguments to spawn child
    std::vector<std::string> args;
    std::vector<std::string> targs;
    std::vector<char *> raw_args;
    {
        std::ostringstream oss;
        args.push_back(cfg.program_name);

        if (cfg.repeats != 1) { args.push_back("--repeat"); oss << cfg.repeats; args.push_back(oss.str()); oss.str(std::string()); };

        if (cfg.random_seed != -1) { args.push_back("--seed"); oss << cfg.random_seed; args.push_back(oss.str()); oss.str(std::string()); };

        args.push_back("--data");

        args.push_back(cfg.data_folder);

        if (cfg.verbose) { args.push_back("--verbose"); }

        if (cfg.suppress_color) { args.push_back("--suppress_color"); }

        args.push_back("--child");
        targs = args;

        for (unsigned int i = 0; i < targs.size(); i++)
        {
            raw_args.push_back(const_cast<char *>(targs[i].c_str()));
        }

        raw_args.push_back((char *)0);
    }

    while (processed < total)
    {
        printf("Spawning new worker\n");
        fflush(stdout);
        int ret_code = -1;
        // spawn and wait for a worker to finish
        //@TODO: add timeouts (ez)
#ifdef _WIN32
        STARTUPINFO si;
        PROCESS_INFORMATION pi;
        std::string long_param;

        for (unsigned int i = 0; i < targs.size(); i++)
        {
            long_param += targs[i] + " ";
        }

        ZeroMemory( &si, sizeof(si) );
        si.cb = sizeof(si);
        ZeroMemory( &pi, sizeof(pi) );

        if ( !CreateProcess( args[0].c_str(),  // No module name (use command line)
                             const_cast<LPSTR>(long_param.c_str()),        // Command line
                             NULL,           // Process handle not inheritable
                             NULL,           // Thread handle not inheritable
                             FALSE,          // Set handle inheritance to FALSE
                             0,              // No creation flags
                             NULL,           // Use parent's environment block
                             NULL,           // Use parent's starting directory
                             &si,            // Pointer to STARTUPINFO structure
                             &pi )           // Pointer to PROCESS_INFORMATION structure
           )
        {
            std::cout << "[SYSTEM ERROR] Cannot spawn unit test Win32 process" << std::endl;
            return 1;
        }

        // Wait until child process exits.
        DWORD win_ret_code;

        if (!WaitForSingleObject( pi.hProcess, INFINITE ))
            if (GetExitCodeProcess(pi.hProcess, &win_ret_code))
            {
                ret_code = win_ret_code;
            }

        // Close process and thread handles.
        CloseHandle( pi.hProcess );
        CloseHandle( pi.hThread );
#else
        int local_ret_code;
        int phandle = fork();

        if (phandle == 0)
        {
            local_ret_code = execv (raw_args[0], &(raw_args[0]));
            _exit(local_ret_code);
        }

        pid_t ws = waitpid( phandle, &local_ret_code, 0);

        if ( !WIFEXITED(local_ret_code) )
        {
            ret_code = 1;
        }
        else if (WIFSIGNALED(local_ret_code))
        {
            ret_code = 1;
        }
        else
        {
            ret_code = local_ret_code;
        }

#endif // win32

        // Child returns nonzero only in the case of emergency exit which means last test fail.
        // We assume that child haven't written fail status to log file, launched file and
        // console output doe to this emergency exit
        // Actually this assuming is not always true.

        // segfault or something
        if (ret_code > 0 && ret_code < 512)
        {
            std::cout << "[" << COLOR_RED << "FAILED" << COLOR_NONE << "]\n";
            std::cout.flush();
            std::ofstream fulog(f_assert_log.c_str(), std::ios_base::app);
            fulog << "[SYSTEM ERROR] Cannot find unit test log. Looks like unit test hasn't finished properly" << std::endl;
            fulog.close();
            // check if last test was launched for the first time?
            std::ifstream last_launched_file(f_secondtry.c_str(), std::ios_base::in);

            if (!last_launched_file.is_open())
            {
                // if not, just report as failed
                std::ofstream fuproc(f_worker_processed.c_str(), std::ios_base::app);
                fuproc << std::endl << 1 << std::endl;
                fuproc.close();
                std::ifstream second_try_file_name(f_secondtry_name.c_str(), std::ios_base::in);
                std::string name, mode;
                std::getline(second_try_file_name, name);
                std::getline(second_try_file_name, mode);
                second_try_file_name.close();
                remove(f_secondtry_name.c_str());
            }
            else
            {
                // if yes - launch for the second time
                //std::string name, mode;
                //std::getline(last_launched_file, name);
                //std::getline(last_launched_file, mode);
                last_launched_file.close();
            }
        }
        // manual exit()
        else if (ret_code == 512)
        {
            //do nothing, process made everything by himself
        }

        // peek into the processing log to count processed unit tests
        std::ifstream fuproc(f_worker_processed.c_str(), std::ios_base::in);
        int cur_processed = 0;
        failed = 0;
        {
            std::string t1;
            int tres;

            while (!fuproc.eof())
            {
                std::getline(fuproc, t1);

                if (fuproc.eof())
                {
                    break;
                }

                std::getline(fuproc, t1);
                fuproc >> tres;

                if (tres != 0) { failed++; }

                std::getline(fuproc, t1);
                cur_processed ++;
            }
        }
        fuproc.close();
        processed = cur_processed;
    }

    // print all asserts:
    std::ifstream fulog(f_assert_log.c_str(), std::ios_base::in);
    std::string line;
    std::cout << "\n---------------------------------------------------------\n"
              "\nUnit test logs:\n";

    if (fulog.is_open())
    {
        while (!fulog.eof())
        {
            std::getline(fulog, line);
            std::cout << line << std::endl;
        }
    }
    else
    {
        std::cout << "Cannot retrieve tests logs" << std::endl;
    }

    std::cout << std::endl;
    printf("Total tests run: %d\t Failed tests: %d\n", total, failed);
    // clear temp stuff
    remove(f_schedule.c_str());
    remove(f_assert_log.c_str());
    remove(f_secondtry.c_str());
    remove(f_secondtry_name.c_str());
    return 0;
}

int UnitTestDriverFramework::run_all_tests(int &total, int &failed, const std::vector<std::string> &modes, const std::vector<std::string> &kwords, const std::vector<std::string> &params)
{
    std::vector<std::string> names_vector;

    for (std::set<std::string>::iterator iter = all_test_names.begin(); iter != all_test_names.end(); ++iter)
    {
        names_vector.push_back(*iter);
    }

    return run_tests(total, failed, modes, kwords, names_vector, params);
}

UnitTestDriverFramework &UnitTestDriverFramework::framework()
{
    static UnitTestDriverFramework s_instance;
    return s_instance;
}

} // end namespace
