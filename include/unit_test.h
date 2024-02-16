// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdio>
#include <string>
#include <vector>
#include "basic_types.h"
#include "matrix_io.h"
#include "util.h"
#include "test_utils.h"
#include "amgx_config.h"
#include <stdarg.h>
#include <iostream>
#include <sstream>
#include <set>


namespace amgx
{

struct UnitTestConfiguration
{
    int       random_seed;
    int       repeats;
    std::string  data_folder;
    std::string  log_file;
    std::string  program_name;
    bool      is_child;
    bool      verbose;
    bool      suppress_color;
    bool      mpi_work;

    UnitTestConfiguration() : random_seed(-1), repeats(1), data_folder(std::string("../test_data/")), log_file(std::string()), is_child(false), verbose(false), suppress_color(false), mpi_work(false)  {};
};

class UnitTestFailedException
{
};

class UnitTestDriverFramework;

class UnitTest
{
    protected:

        std::string _name;
        bool        _failed;
        bool        _failed_bit;
        bool        _forge_ahead;
        int         _last_seed;
        int         _random_seed;
        char        err_buffer[1024];
        AMGX_Mode  _mode;
        std::stringstream _assert_ss;
        std::stringstream _err_stream;

        std::string keywords;
        std::string custom_cl;
        UnitTest(const char *name, AMGX_Mode mode, const char *kwords = NULL);

        void set_failed() { _failed = true; _failed_bit = true; }

        void reset_err_buffer() { err_buffer[0] = 0; _err_stream.str(std::string());}

        void PrintOnFail(const char *fmt, ...)
        {
            va_list args;
            va_start(args, fmt);
            vsnprintf(err_buffer, 1024, fmt, args);
            _err_stream << err_buffer;
            va_end(args);
        }

        template<class TConfig>
        bool read_system(const char *path, Matrix<TConfig> &A, Vector<TConfig> &b, Vector<TConfig> &x, bool abs_path = false)
        {
            AMG_Config t_cfg;
            std::string full_path = get_configuration().data_folder + std::string(path);
            return (MatrixIO<TConfig>::readSystem(abs_path ? path : full_path.c_str(), A, b, x, t_cfg) == AMGX_OK);
        }

        void randomize(int seed);

        template<class Container>
        void random_fill(Container &x)
        {
            fillRandom<Container>::fill(x);
        }

        template<class Container1, class Container2>
        bool check_hashsum(const Container1 &x1, const Container2 &x2)
        {
            return check_hashsum_equal<Container1, Container2>::check(x1, x2);
        }

        void add_keywords(const std::string &new_kwords)
        {
            if (new_kwords.length() > 0)
            {
                if (keywords.length() > 0)
                {
                    keywords += ';';
                }

                keywords += new_kwords;
            }
        }

        void set_command_line(const std::string &new_cl)
        {
            custom_cl = new_cl;
        }

// control funtions
        virtual void run() = 0;
        virtual void start() {};
        virtual void end() {};

    public:

        static bool amgx_intialized;

        static UnitTestConfiguration &get_configuration();

        const std::string &name() const { return _name; }
        const std::string &getKeywords() const { return keywords; }
        const std::string &getCommandLine() const { return custom_cl; }
        AMGX_Mode getMode() const { return _mode; }


        virtual std::string base_keywords();
        virtual std::string custom_launch_line();

        void reset()
        {
            reset_err_buffer();
            _failed = false;
            _failed_bit = false;
        }

        bool failed() const { return _failed; }
        void set_forge_ahead(bool new_val) { _forge_ahead = new_val; }

        void print_assert(std::ostream &str) const { str << _assert_ss.str(); }



// asserts functions
        template <class ValueType1, class ValueType2, class ValueType3 >
        void assert_equal_tol ( ValueType1 a, ValueType2 b, ValueType3 tol, const char *filename, int lineno)
        {
            std::stringstream buffer;

            if (!check_equal_tolerance<ValueType1, ValueType2, ValueType3>::check(a, b, tol, buffer))
            {
                _assert_ss << "[ASSERT] " << " in test " << _name.c_str() << " at " << filename << " lineno: " << lineno << " last rand seed: " << _last_seed << std::endl << "Comparison info: " << buffer.str() << std::endl;

                if (_err_stream.str().length()) { _assert_ss <<  _err_stream.str(); }

                set_failed();

                if (!_forge_ahead)
                {
                    throw UnitTestFailedException();
                }
            }

            reset_err_buffer();
        }

        template <class ValueType1, class ValueType2 >
        void assert_equal ( bool flag, ValueType1 a, ValueType2 b, const char *filename, int lineno)
        {
            std::stringstream buffer;

            if (check_equal<ValueType1, ValueType2>::check(a, b, buffer) != flag)
            {
                _assert_ss << "[ASSERT] " << " in test " << _name.c_str() << " at " << filename << " lineno: " << lineno << " last rand seed: " << _last_seed << std::endl << "Comparison info: " << buffer.str() << std::endl;

                if (_err_stream.str().length()) { _assert_ss <<  _err_stream.str(); }

                set_failed();

                if (!_forge_ahead)
                {
                    throw UnitTestFailedException();
                }
            }

            reset_err_buffer();
        }
        void assert_finite( double, const char *filename, int lineno);
        void assert_true  ( bool, const char *filename, int lineno);
        void assert_never ( const char *filename, int lineno);

        int start_test();

};

#define UNITTEST_ASSERT_EQUAL( a, b )                           this->assert_equal(true, (a), (b), __FILE__, __LINE__)
#define UNITTEST_ASSERT_EQUAL_DESC( desc, a, b )                this->PrintOnFail(desc); this->assert_equal(true, (a), (b), __FILE__, __LINE__)

#define UNITTEST_ASSERT_NEQUAL( a, b )                          this->assert_equal(false, (a), (b), __FILE__, __LINE__)
#define UNITTEST_ASSERT_NEQUAL_DESC( desc, a, b )               this->PrintOnFail(desc); this->assert_equal(false, (a), (b), __FILE__, __LINE__)

#define UNITTEST_ASSERT_EQUAL_TOL( a, b ,tolerance )            this->assert_equal_tol((a), (b), (tolerance),__FILE__, __LINE__)
#define UNITTEST_ASSERT_EQUAL_TOL_DESC( desc, a, b ,tolerance ) this->PrintOnFail(desc); this->assert_equal_tol((a), (b), (tolerance),__FILE__, __LINE__)

#define UNITTEST_ASSERT_FINITE(a)                               this->assert_finite((a), __FILE__, __LINE__)
#define UNITTEST_ASSERT_FINITE_DESC(desc, a)                    this->PrintOnFail(desc); this->assert_finite((a), __FILE__, __LINE__)

#define UNITTEST_ASSERT_TRUE(a)                                 this->assert_true((a), __FILE__, __LINE__)
#define UNITTEST_ASSERT_TRUE_DESC(desc, a)                      this->PrintOnFail(desc); this->assert_true((a), __FILE__, __LINE__)

#define UNITTEST_ASSERT_EXCEPTION_START                         {bool _c = true; _c = _c; try {
#define UNITTEST_ASSERT_EXCEPTION_END_ANY                       } catch(...) {_c=false;} if (_c) this->assert_never(__FILE__, __LINE__); }
#define UNITTEST_ASSERT_EXCEPTION_END_ANY_DESC(desc)            } catch(...) {_c=false;} if (_c) {this->PrintOnFail(desc);this->assert_never(__FILE__, __LINE__);} }
#define UNITTEST_ASSERT_EXCEPTION_END(e)                        } catch( e & _e) {_c=false;} if (_c) this->assert_never(__FILE__, __LINE__); }
#define UNITTEST_ASSERT_EXCEPTION_END_DESC(desc, e)             } catch( e & _e) {_c=false;} if (_c) {this->PrintOnFail(desc);this->assert_never(__FILE__, __LINE__);} }
#define UNITTEST_ASSERT_EXCEPTION_END_AMGX_ERR(e)              } catch( amgx_exception & _e) { if ( _e.reason() != e ) throw; _c = false;} if (_c) this->assert_never(__FILE__, __LINE__); }
#define UNITTEST_ASSERT_EXCEPTION_END_AMGX_ERR_DESC(desc, e)   } catch( amgx_exception & _e) { if ( _e.reason() != e ) throw; _c = false;} if (_c) {this->PrintOnFail(desc);this->assert_never(__FILE__, __LINE__);} }
#define UNITTEST_ASSERT_EXCEPTION_END_NOT_IMPLEMENTED           } catch( amgx_exception & _e) { if (_e.reason() != AMGX_ERR_NOT_SUPPORTED_TARGET && _e.reason() != AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE &&  \
                                                                                                     _e.reason() != AMGX_ERR_NOT_IMPLEMENTED) throw; } }

#define ONLY_IF_SUPPORTED( Line ) \
  UNITTEST_ASSERT_EXCEPTION_START \
  Line;                           \
  UNITTEST_ASSERT_EXCEPTION_END_NOT_IMPLEMENTED


#define DECLARE_UNITTEST_BEGIN_EXTD(TestClass, BaseClass) \
  template<class T_Config> \
  class TestClass : public BaseClass { \
  public: \
    typedef T_Config TConfig;\
    typedef TemplateConfig<AMGX_host, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_h;\
    typedef TemplateConfig<AMGX_device, TConfig::vecPrec, TConfig::matPrec, TConfig::indPrec> TConfig_d;\
    typedef typename T_Config::MatPrec ValueTypeA;\
    typedef typename T_Config::VecPrec ValueTypeB;\
    typedef Vector<T_Config> VVector;\
    typedef typename Matrix<TConfig>::MVector MVector;\
    typedef typename Matrix<TConfig_h>::MVector MVector_h;\
    typedef Matrix<T_Config> MatrixA;\
    typedef Matrix< TConfig_h > Matrix_h;\
    typedef Matrix< TConfig_d > Matrix_d;\
    typedef Vector< TConfig_h > Vector_h;\
    typedef Vector< TConfig_d > Vector_d;\
    typedef typename Matrix<T_Config>::IVector IVector;\
    typedef typename Matrix<TConfig_h>::IVector IVector_h;\
    typedef typename T_Config::IndPrec IndexType;\
    TestClass(const char* kwords = NULL) : BaseClass(#TestClass, static_cast<AMGX_Mode>(TConfig::mode), kwords) { this->add_keywords(this->base_keywords()); this->set_command_line(this->custom_launch_line()); }\
    TestClass(const char *name, AMGX_Mode mode, const char* kwords = NULL) : BaseClass(name, mode, kwords) { this->add_keywords(this->base_keywords()); this->set_command_line(this->custom_launch_line());}\


#define DECLARE_UNITTEST_BEGIN(TestClass) DECLARE_UNITTEST_BEGIN_EXTD(TestClass, UnitTest)
#define DECLARE_UNITTEST_END(TEST) \
  };


} // end namespace

