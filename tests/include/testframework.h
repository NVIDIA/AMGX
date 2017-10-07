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

#include "unit_test.h"

// contains all combinations test + mode that hasn't been launched yet
// file written by host process and readed by child to udnerstand what tests to run
const std::string f_schedule            = std::string(".schedule");

// list of already launched tests
// written by child, appends after each test run. readed by host to create new schedule in case of crash of child
const std::string f_worker_processed    = std::string(".launched");

// tests log
const std::string f_assert_log          = std::string(".utestlog");

// unit tests results in numbers - number of failed, successful and overall tests
const std::string f_statfile            = std::string(".utestsnums");

// child creates this file each time before first launch of the test and removed by the end of the test
// if test crashes and file still here - we remove file and launch test for the second time
const std::string f_secondtry           = std::string(".secondtryflag");

// this file will contain last launched test in the case of crash of the second launched.
const std::string f_secondtry_name      = std::string(".secondtryname");

namespace amgx
{

class UnitTestDriver
{

        friend class UnitTestDriverFramework;
        std::map<std::string, std::map<std::string, UnitTest *> >  tests;
        std::pair<std::string, std::string>                       last_launched;
        bool                                                      check_last_launch;

    public:

        UnitTestDriver() { check_last_launch = false; };

// register new test
        void register_test(UnitTest *test);

// -1 = no such test
// 0  = OK
// 1  = failed
        int run_test  (const std::string &name, const std::string &mode, std::ostream &logfile);

        int run_tests (const std::vector< std::pair<std::string, std::string> > &tests, std::ostream &logfile);

// NULL if no such test
        UnitTest *get_test (const std::string &name, const std::string &mode) const;

// prints all tests like this:
// TestName: supported_mode1 supported_mode2 ....
        void print_all_tests(std::ostream &str);
};


class UnitTestDriverFramework
{
        std::set<std::string>                     all_test_names;
        UnitTestDriver                            test_driver;
        std::map<std::string, int>                kwords_dict;

        UnitTestConfiguration                     configuration;

    public:

        UnitTestDriverFramework() {};
        ~UnitTestDriverFramework();

        void register_test(UnitTest *test);

        // used for top-level calls
        int run_tests    (int &total, int &failed, const std::vector<std::string> &modes, const std::vector<std::string> &kwords, const std::vector<std::string> &tests, const std::vector<std::string> &params);
        int run_all_tests(int &total, int &failed, const std::vector<std::string> &modes, const std::vector<std::string> &kwords, const std::vector<std::string> &params);

        // called only if child worker
        void do_work();


        // util
        const std::set<std::string> &get_all_names() const {return all_test_names;}
        //const UnitTestConfiguration& get_configuration() const {return configuration;}
        void set_configuration(UnitTestConfiguration &cfg) {UnitTest::get_configuration() = cfg;}
        int keyword_register(std::string &kword)
        {
            if (kwords_dict.find(kword) == kwords_dict.end())
            {
                kwords_dict[kword] = kwords_dict.size();
            }

            return kwords_dict[kword];
        }
        bool keyword_exists(std::string &kword)
        {
            return (kwords_dict.find(kword) != kwords_dict.end());
        }

        static UnitTestDriverFramework &framework();
};

} // end namespace

