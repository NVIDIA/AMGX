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

#include <map>
#include <string>
#include <vector>
#include <string.h>  //strtok
#include <string>
#include <typeinfo>
#include <error.h>
#include <amg_signal.h>

#ifdef RAPIDJSON_DEFINED
#include "rapidjson/document.h"
#endif

namespace amgx
{

/******************************************
 * A class for storing a typeless parameter
 *****************************************/
class Parameter
{
    public:
        Parameter() { memset(data, 0, 64); }
        template<typename T> Parameter(T value)
        {
            set(value);
        }
        //return the parameter as the templated type
        template<typename T> T get() const;

        //set the parameter from the templated type
        template<typename T> void set(T value);

    private:
        char data[64]; //64 bytes of storage
};

//return the parameter as a string
template<> inline std::string Parameter::get() const
{
    return std::string(data);
}
//set the parameter from a string
template<> inline void Parameter::set(std::string value)
{
    if (value.length() > 64)
    {
        std::string error("Parameter '" + value + "' can only be 64 characters in length");
        FatalError(error.c_str(), AMGX_ERR_CONFIGURATION);
    }

    strncpy(data, value.c_str(), 64);
}
//return the parameter as the templated type
template<typename T> inline T Parameter::get() const
{
    T value = *reinterpret_cast<const T *>(&data[0]);
    return value;
}
//set the parameter from the templated type
template<typename T> inline void Parameter::set(T value)
{
    if (sizeof(T) > 64)
    {
        FatalError("Parameter size is not large enough", AMGX_ERR_CONFIGURATION);
    }

    *reinterpret_cast<T *>(&data[0]) = value;
}

/*******************************************
 * A class to store possible parameter values
 ******************************************/
typedef enum {PM_NOT_SET = 0, PM_SET = 1, PM_MINMAX = 2} ParameterMarginsType;

class ParameterMargins
{
    public:
        ParameterMargins() : pm_type(PM_NOT_SET) {}

        template<typename T>
        ParameterMargins(const std::vector<T> &val) : pm_type(PM_SET)
        {
            value_set.clear();

            for (int i = 0; i < val.size(); i++)
            {
                value_set.push_back(Parameter(val[i]));
            }
        }
        ParameterMargins(const Parameter &min, const Parameter &max) : pm_type(PM_MINMAX), value_min_max(std::make_pair(min, max)) {}
        ParameterMarginsType pm_type;
        std::vector<Parameter> value_set;                   // set of possible parameter values
        std::pair<Parameter, Parameter> value_min_max;      // pair of minimum and maximum parameter value for continuous parameter space
};

/*******************************************
 * A class to store a description of a
 * parameter
 ******************************************/
class ParameterDescription
{
    public:
        ParameterDescription() : type(0) {}
        ParameterDescription(const ParameterDescription &p) : type(p.type), name(p.name), description(p.description), default_value(default_value), allowed_values(p.allowed_values) {}
        ParameterDescription(const std::type_info *type, const std::string &name, const std::string &description, const Parameter &default_value) : type(type), name(name), description(description), default_value(default_value), allowed_values(ParameterMargins()) {}
        ParameterDescription(const std::type_info *type, const std::string &name, const std::string &description, const Parameter &default_value, const ParameterMargins &parameter_allowed_values) : type(type), name(name), description(description), default_value(default_value), allowed_values(parameter_allowed_values) {}
        ParameterDescription& operator=(const ParameterDescription&) = default;
        mutable const std::type_info *type;   //the type of the parameter
        std::string name;             //the name of the parameter
        std::string description;      //description of the parameter
        Parameter default_value;      //the default value of the parameter
        ParameterMargins allowed_values; // possible parameter values
};


/***********************************************
 * A class for storing paramaters in a database
 * which includes type information.
 **********************************************/
class AMG_Config
{

    private:
        typedef std::map<std::string, ParameterDescription> ParamDesc;

        static ParamDesc param_desc;  //The parameter descriptions


    public:
        AMG_Config();
        /***********************************************
         * Registers the parameter in the database.
        **********************************************/
        template <typename Type> static void registerParameter(std::string name, std::string description, Type default_value)
        {
            param_desc[name] = ParameterDescription(&typeid(Type), name, description, default_value);
        }

        template <typename Type> static void registerParameter(std::string name, std::string description, Type default_value, const Type min_value, const Type max_value)
        {
            param_desc[name] = ParameterDescription(&typeid(Type), name, description, default_value, ParameterMargins(Parameter(min_value), Parameter(max_value)));
        }

        template <typename Type> static void registerParameter(std::string name, std::string description, Type default_value, const std::vector<Type> &allowed_values)
        {
            param_desc[name] = ParameterDescription(&typeid(Type), name, description, default_value, ParameterMargins(allowed_values));
        }

        /***********************************************
         * Unregisters the parameter in the database by its key value.
        **********************************************/
        static void unregisterParameter(std::string name)
        {
            param_desc.erase(param_desc.find(name));
        }

        static void unregisterParameters()
        {
            param_desc.clear();
        }

        static std::string getParamTypeName(const std::type_info *param_type);

        /********************************************
        * Gets a parameter from the database and
        * throws an exception if it does not exist.
        *********************************************/
        template <typename Type> Type getParameter(const std::string &name, const std::string &current_scope) const;
        template <typename Type> void getParameter(const std::string &name, Type &value, const std::string &current_scope, std::string &new_scope) const;

        AMGX_ERROR parseParameterString(const char *str);

        AMGX_ERROR parseParameterStringAndFile(const char *str, const char *filename);

        template<typename T>
        void setNamedParameter(const std::string &name, const T &c_value, const std::string &current_scope, const std::string &new_scope, ParamDesc::iterator &param_desc_iter);
        template<typename T>
        void importNamedParameter(const char *c_name, const T &c_value, const std::string &current_scope, const std::string &new_scope);

#ifdef RAPIDJSON_DEFINED
        AMGX_ERROR parse_json_file(const char *filename);
        AMGX_ERROR parse_json_string(const char *str);
        void import_json_object(rapidjson::Value &obj, bool outer);
#endif

        // this will return an error if JSON is not supported
        static AMGX_ERROR write_parameters_description_json(const char *filename);

        /****************************************************
        * Parse a config file
        ****************************************************/
        AMGX_ERROR parseFile(const char *filename);

        /**********************************************
        * Sets a parameter in the database
        * throws an exception if it does not exist.
        *********************************************/
        template <typename Type> void setParameter(std::string name, Type value, const std::string &current_scope, const std::string &new_scope);
        template <typename Type> void setParameter(std::string name, Type value, const std::string &current_scope);

        /****************************************************
         * Print the options for AMG
         ***************************************************/
        static void printOptions();

        /***************************************************
         * Prints the AMG parameters                       *
         ***************************************************/
        void printAMGConfig();

        /***************************************************
         * Convert a parameter value to a string
         * ************************************************/
        std::string getParameterString(Parameter &parameter, ParameterDescription &param_desc);

        void setAllowConfigurationMod(int flag) { m_allow_cfg_mod = !(!flag);}
        int  getAllowConfigurationMod() { return m_allow_cfg_mod; }

        int ref_count;

    private:
        static SignalHandler sh;  //install the signal handlers here

        typedef std::map< std::pair<std::string, std::string>, std::pair<std::string, Parameter> > ParamDB;

        ParamDB m_params;               //The parameter database
        std::vector<std::string> m_scope_vector;
        std::vector<std::string> m_solver_list;
        int m_latest_config_version;
        int m_config_version;

        int m_allow_cfg_mod;

        /***************************************************************************
         * Extract the name, value, current_scope and new_scope of a single entry
         * ************************************************************************/
        void extractParamInfo(const std::string &str, std::string &name, std::string &value, std::string &current_scope, std::string &new_scope);


        /****************************************************
        * Parse a string in the format
        * name=value
        * and store the variable in the parameter database
        ****************************************************/
        void setParameter(const std::string &str);

        void getOneParameterLine(std::string &params, std::string &param, int &idx);

        int getConfigVersion(std::string &params);

        void convertToCurrentConfigVersion(std::string &params, int config_version);

        /****************************************************
        * Convert a config file to a config string
        ****************************************************/
        AMGX_ERROR getParameterStringFromFile(const char *filename, std::string &params);

        AMGX_ERROR checkString(std::string &str);

        /****************************************************
         * Parse parameters in the format
         * scope:name(new_scope)=value, scope:name(new_scope)=value, ... scope:name(new_scope)=value
         * and store the variables in the parameter database
         ****************************************************/
        AMGX_ERROR parseString(std::string &str, int &config_version);

        void clear();
};

template <class T_Config> class AMG_Solver;

class AMG_Configuration
{
        template <class> friend class AMG_Solver;
    public:
        AMG_Configuration();
        AMG_Configuration(const AMG_Configuration &cfg);
        AMG_Configuration &operator=(const AMG_Configuration &cfg);
        ~AMG_Configuration();

        /********************************************
         * Gets a parameter from the database and
         * throws an exception if it does not exist.
         *********************************************/
        template <typename Type> Type getParameter(std::string name) const;

        /**********************************************
         * Sets a parameter in the database
         * throws an exception if it does not exist.
         *********************************************/
        template <typename Type> void setParameter(std::string name, Type value, const std::string &current_scope);

        /****************************************************
         * Parse paramters in the format
         * config_version=value, scope:name(new_scope)=value,scope:name(new_scope)=value, ...,scope:name(new_scope)=value
         * and store the variables in the parameter database
         ****************************************************/
        AMGX_ERROR parseParameterString(const char *str);

        /****************************************************
        * Parse a config file  in the format
        * config_version=value
        * scope:name(new_scope)=value
        * scope:name(new_scope)=value
        * ...
        * scope:name(new_scope)=value
        * and store the variables in the parameter database
        ****************************************************/

        AMGX_ERROR parseFile(const char *filename);

        /*****************************************
         * Parse a config file and a config string
         *****************************************/

        AMGX_ERROR parseParameterStringAndFile(const char *str, const char *filename);

        /****************************************************
         * Print the options for AMG
         ***************************************************/
        static void printOptions();

        void setAllowConfigurationMod(int flag) { amg_config->setAllowConfigurationMod(flag); }


        /// The configuration object to call factories from outside AMG_Solver.`
        inline AMG_Config *getConfigObject() { return amg_config; }

    private:
        AMG_Config *amg_config;
        int ref_count;
};

} // namespace amgx
