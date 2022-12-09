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

#include <amg_solver.h>
#include <amg_config.h>
#include <iostream>
#include <error.h>
#include <fstream>
#include <types.h>
#include <util.h>
#include <misc.h>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <device_properties.h>

#ifdef RAPIDJSON_DEFINED
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/filestream.h"
#endif

namespace amgx
{

#ifdef RAPIDJSON_DEFINED
static int unnamed_scope_counter = 0;
static rapidjson::Document json_parser; //made as global to avoid passing template parameter memory allocator via parameters
#endif

// trim from start
static inline std::string &ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}

// trim from end
static inline std::string &rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}

// trim from both ends
static inline std::string &trim(std::string &s)
{
    return ltrim(rtrim(s));
}

AMG_Config::ParamDesc AMG_Config::param_desc;

__inline__ bool allowed_symbol(const char &a)
{
    return (a >= 'a' && a <= 'z') || (a >= 'A' && a <= 'Z') || (a >= '0' && a <= '9') || (a == '=') || (a == '_') || (a == '.') || (a == '-') || (a == '+');
}


// Parses the supplied parameter string
AMGX_ERROR AMG_Config::parseParameterString(const char *str)
{
    // rapidjson doesn't handle NULL
    if (!str) return AMGX_ERR_CONFIGURATION;
    
    try
    {
#ifdef RAPIDJSON_DEFINED
        if (parse_json_string(str) != AMGX_OK)
        {
#endif
            //copy to a temporary array to avoid destroying the string
            std::string params(str);
            // Read the config version
            int config_version = getConfigVersion(params);

            // Parse the parameter string
            if (parseString(params, config_version) != AMGX_OK)
            {
                std::string err = "Error parsing parameter string: " + params;
                FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
            }

#ifdef RAPIDJSON_DEFINED
        }

#endif
    }
    catch (amgx_exception &e)
    {
        amgx_printf("Error parsing parameter string: %s\n", e.what());
        return e.reason();
    }
    catch (...)
    {
        amgx_printf("Error parsing parameter string\n");
        return AMGX_ERR_CONFIGURATION;
    }

    return AMGX_OK;
}

// Parses the supplied parameter string
AMGX_ERROR AMG_Config::parseParameterStringAndFile(const char *str, const char *filename)
{
    AMGX_ERROR ret = AMGX_OK;

    try
    {
        ret = parseFile(filename);

        if (ret == AMGX_OK)
        {
            ret = parseParameterString(str);
        }
    }
    catch (amgx_exception &e)
    {
        amgx_printf("Error parsing parameter string: %s\n", e.what());
        ret = e.reason();
    }
    catch (...)
    {
        amgx_printf("Error parsing parameter string\n");
        ret = AMGX_ERR_CONFIGURATION;
    }

    return ret;
}



// Extracts a single parameter line from the config string and increments idx
void AMG_Config::getOneParameterLine(std::string &params, std::string &param, int &idx)
{
    param.erase();

    while ((idx < params.length()) && (params[idx] != ',') && (params[idx] != ';')) // config delimiters
    {
        param += params[idx];
        idx ++;
    }

    idx++;
}

// Gets an input config_string, gets the config_version value and removes the config_version entry from the file
int AMG_Config::getConfigVersion(std::string &params)
{
    int idx = 0;
    std::string param;
    getOneParameterLine(params, param, idx);

    if (param.length() > 2 && param.find_first_not_of(' ') != std::string::npos) /* check that param is not empty and check length: one for parameter name, one for the equal sign, one for the parameter value. otherwise - this is error */
    {
        // Extract the name, value, current_scope, new_scope
        std::string name, value, current_scope, new_scope;
        extractParamInfo(param, name, value, current_scope, new_scope);
        int config_version;

        if (name == "config_version")
        {
            config_version = getValue<int>(value.c_str());

            if (config_version != 1 && config_version != 2)
            {
                std::string err = "Error, config_version must be 1 or 2. Config string is " + std::string(param);
                FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
            }

            // Erase the config_version entry
            params.erase(0, idx);
        }
        else
        {
            config_version = 1;
        }

        return config_version;
    }

    return 1;
}

void AMG_Config::convertToCurrentConfigVersion(std::string &params, int config_version)
{
    // --------------------------
    // DO THE CONVERSION HERE
    // --------------------------
    std::string old_params = params;
    params.erase();

    if (config_version == 1)
    {
        int idx = 0;
        std::string param;

        while (idx < old_params.length())
        {
            getOneParameterLine(old_params, param, idx);

            if (param.length() > 2 && param.find_first_not_of(' ') != std::string::npos) /* check that param is not empty and check length: one for parameter name, one for the equal sign, one for the parameter value. otherwise - this is error */
            {
                std::string name, value, current_scope, new_scope;
                extractParamInfo(param, name, value, current_scope, new_scope);

                if (current_scope != "default" || new_scope != "default")
                {
                    std::string err = "Error parsing parameter string: " + param + " . Scopes only supported with config_version=2 and higher. Add \"config_version=2\" to the config string to use nested solvers";
                    FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
                }
                //else if (name == "smoother"  )
                //{
                // Add coloring if necessary
                //if (value=="MULTICOLOR_DILU" || value=="MULTICOLOR_GS" || value=="MULTICOLOR_ILU")
                //{
                //  params += "coloring_level=1 ; ";
                //}
                //params += ( name + "=" + value + " ; ");
                //}
                else if (name == "smoother_weight")
                {
                    params += "relaxation_factor=" + value + " ; ";
                }
                else if (name == "min_block_rows")
                {
                    params += "min_coarse_rows=" + value + " ; ";
                }
                else if (value == "JACOBI" || value == "JACOBI_NO_CUSP")
                {
                    params += name + "=" + "BLOCK_JACOBI ; ";
                }
                else
                {
                    // Just add it to default scope
                    params += ( name + "=" + value + " ; ");
                }
            }
        }
    }
    else if (config_version == 2)
    {
    }
    else
    {
        std::stringstream err;
        err << "Invalid config_version (config_version must be 1 or 2)" << std::endl;
        FatalError(err.str().c_str(), AMGX_ERR_CONFIGURATION);
    }
}

// Parses the supplied parameter string
AMGX_ERROR AMG_Config::parseString(std::string &params, int &config_version)
{
    try
    {
        m_config_version = config_version;

        if (m_config_version != this->m_latest_config_version)
        {
            size_t pos = params.find("verbosity_level");
            pos = params.find("=", pos);
            pos++;

            while (params[pos] == ' ') { pos++; }  // skip spaces

            bool print = (pos == std::string::npos) || (params[pos] > '2');
            std::string ss = "Converting config string to current config version\n";

            if (print)
            {
#ifdef AMGX_WITH_MPI
                amgx_distributed_output(ss.c_str(), ss.length());
#else
                amgx_output(ss.c_str(), ss.length());
#endif
            }

            convertToCurrentConfigVersion(params, config_version);
            ss = "Parsing configuration string: " + params + "\n";

            if (print)
            {
#ifdef AMGX_WITH_MPI
                amgx_distributed_output(ss.c_str(), ss.length());
#else
                amgx_output(ss.c_str(), ss.length());
#endif
            }
        }

        // Parse the individual parameters
        int idx = 0;
        std::string param;

        while (idx < params.length())
        {
            getOneParameterLine(params, param, idx);

            if (param.length() > 2 && param.find_first_not_of(' ') != std::string::npos) /* check that param is not empty and check length: one for parameter name, one for the equal sign, one for the parameter value. otherwise - this is error */
            {
                setParameter(param);
            }
        }
    }
    catch (amgx_exception &e)
    {
        amgx_printf("Error parsing parameter string: %s\n", e.what());
        return e.reason();
    }
    catch (...)
    {
        amgx_printf("Error parsing parameter string\n");
        return AMGX_ERR_CONFIGURATION;
    }

    return AMGX_OK;
}

AMGX_ERROR AMG_Config::getParameterStringFromFile(const char *filename, std::string &params)
{
    std::ifstream fin;

    try
    {
        // Store the file content into a string
        params = "";
        fin.open(filename);

        if (!fin)
        {
            char error[500];
            sprintf(error, "Error opening file '%s'", filename);
            FatalError(error, AMGX_ERR_IO);
        }

        while (!fin.eof())
        {
            std::string line;
            std::getline(fin, line);
            line = trim(line);

            if (line.empty() || line[0] == '#')
            {
                continue;
            }

            params += (line + ", ");
        }

        fin.close();
    }
    catch (amgx_exception &e)
    {
        amgx_output(e.what(), strlen(e.what()));

        if (fin) { fin.close(); }

        return e.reason();
    }
    catch (...)
    {
        return AMGX_ERR_UNKNOWN;
    }

    return AMGX_OK;
}


std::string AMG_Config::getParamTypeName(const std::type_info *param_type)
{
    // stupid but portable version to avoid demangling
    if (typeid(int) == *param_type)
    {
        return "int";
    }
    else if (typeid(size_t) == *param_type)
    {
        return "size_t";
    }
    else if (typeid(double) == *param_type)
    {
        return "double";
    }
    else if (typeid(std::string) == *param_type)
    {
        return "string";
    }
    else if (typeid(AlgorithmType) == *param_type)
    {
        return "AlgorithmType";
    }
    else if (typeid(ViewType) == *param_type)
    {
        return "ViewType";
    }
    else if (typeid(ColoringType) == *param_type)
    {
        return "ColoringType";
    }
    else if (typeid(BlockFormat) == *param_type)
    {
        return "BlockFormat";
    }
    else if (typeid(NormType) == *param_type)
    {
        return "NormType";
    }

    return "Unknown type";
}

template<typename T>
void AMG_Config::setNamedParameter(const std::string &name, const T &c_value, const std::string &current_scope, const std::string &new_scope, ParamDesc::iterator &param_desc_iter)
{
    std::string err = "Parameter " + name + "(" + current_scope + ") is of unknown type, cannot import value.";
    FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
}

template<>
void AMG_Config::setNamedParameter(const std::string &name, const std::string &c_value, const std::string &current_scope, const std::string &new_scope, ParamDesc::iterator &param_desc_iter)
{
    if (*(param_desc_iter->second.type) == typeid(AlgorithmType))
    {
        setParameter(name, getValue<AlgorithmType>(c_value.c_str()), current_scope, new_scope);
    }
    else if (*(param_desc_iter->second.type) == typeid(ViewType))
    {
        setParameter(name, getValue<ViewType>(c_value.c_str()), current_scope, new_scope);
    }
    else if (*(param_desc_iter->second.type) == typeid(ColoringType))
    {
        setParameter(name, getValue<ColoringType>(c_value.c_str()), current_scope, new_scope);
    }
    else if (*(param_desc_iter->second.type) == typeid(BlockFormat))
    {
        setParameter(name, getValue<BlockFormat>(c_value.c_str()), current_scope, new_scope);
    }
    else if (*(param_desc_iter->second.type) == typeid(NormType))
    {
        setParameter(name, getValue<NormType>(c_value.c_str()), current_scope, new_scope);
    }
    else if (*(param_desc_iter->second.type) == typeid(std::string))
    {
        setParameter(name, c_value, current_scope, new_scope);
    }
    else
    {
        std::string err = "Incorrect config entry. Type of the parameter \"" + name + "\" in the config is string, but " + getParamTypeName(param_desc_iter->second.type) + " is expected";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }
}

template<>
void AMG_Config::setNamedParameter(const std::string &name, const double &c_value, const std::string &current_scope, const std::string &new_scope, ParamDesc::iterator &param_desc_iter)
{
    if (typeid(double) == *(param_desc_iter->second.type))
    {
        setParameter(name, c_value, current_scope, new_scope);
    }
    else if (typeid(int) == *(param_desc_iter->second.type))
    {
        int _i_val = (int)(c_value);
        setParameter(name, _i_val, current_scope, new_scope);
    }
    else
    {
        std::string err = "Incorrect config entry. Type of the parameter \"" + name + "\" in the config is double, but " + getParamTypeName(param_desc_iter->second.type) + " is expected";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }
}

template<>
void AMG_Config::setNamedParameter(const std::string &name, const int &c_value, const std::string &current_scope, const std::string &new_scope, ParamDesc::iterator &param_desc_iter)
{
    if (typeid(int) == *(param_desc_iter->second.type))
    {
        setParameter(name, c_value, current_scope, new_scope);
    }
    else if (typeid(double) == *(param_desc_iter->second.type))
    {
        double _d_val = (double)(c_value);
        setParameter(name, _d_val, current_scope, new_scope);
    }
    else
    {
        std::string err = "Incorrect config entry. Type of the parameter \"" + name + "\" in the config is int, but " + getParamTypeName(param_desc_iter->second.type) + " is expected";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }
}


template<typename T>
void AMG_Config::importNamedParameter(const char *c_name, const T &c_value, const std::string &current_scope, const std::string &new_scope)
{
    std::string name = c_name;

    // Add new_scope to scope vector
    if ( find(m_scope_vector.begin(), m_scope_vector.end(), new_scope) == m_scope_vector.end())
    {
        m_scope_vector.push_back(new_scope);
    }
    else if (new_scope != "default" && !getAllowConfigurationMod())
    {
        std::string err = "Incorrect config entry (new scope already defined): " + new_scope;
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }

    // extract the name, value, new_scope and old_scope
    //verify parameter was registered
    ParamDesc::iterator iter = param_desc.find(std::string(name));

    if (iter == param_desc.end())
    {
        std::string err = "Variable '" + std::string(name) + "' not registered";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }

    if ( (name == "determinism_flag" || name == "block_format" || name == "separation_interior" || name == "separation_exterior" || name == "min_rows_latency_hiding" || name == "fine_level_consolidation" || name == "use_cuda_ipc_consolidation") && current_scope != "default" )
    {
        std::string err = "Incorrect config entry. Parameter " + name + " can only be specified with default scope.";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }

    // Check that new scope is only associated with a solver
    if (new_scope != "default" && find(m_solver_list.begin(), m_solver_list.end(), name) == m_solver_list.end() )
    {
        std::string err = "Incorrect config entry. New scope can only be associated with a solver. new_scope=" + new_scope + ", name=" + name + ".";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }

    // Set the new parameter name, value
    setNamedParameter(name, c_value, current_scope, new_scope, iter);
}

#ifdef RAPIDJSON_DEFINED

void AMG_Config::import_json_object(rapidjson::Value &obj, bool outer)
{
    const char *json_type_names[] = { "Null", "False", "True", "Object", "Array", "String", "Number" };
    std::string current_scope = "default";
    std::string default_new_scope = "default";

    if (obj.HasMember("scope"))
    {
        current_scope = obj["scope"].GetString();
    }

    for (rapidjson::Value::MemberIterator iter = obj.MemberBegin(); iter != obj.MemberEnd(); ++iter)
    {
        if (strcmp(iter->name.GetString(), "config_version") == 0 || strcmp(iter->name.GetString(), "scope") == 0)
        {
            continue;
        }

        if (strcmp(iter->name.GetString(), "solver") == 0 && !outer)
        {
            continue;
        }

        if (strcmp(iter->name.GetString(), "eig_solver") == 0 && !outer)
        {
            continue;
        }

        //printf("Parsing parameter with name \"%s\" of type %s\n", iter->name.GetString(), json_type_names[iter->value.GetType()]);
        if (iter->value.IsObject())
        {
            if (!iter->value.HasMember("scope"))
            {
                char tmp[32];
#ifdef _WIN32
                _snprintf(
#else
                snprintf(
#endif
                    tmp, 31, "unnamed_solver_%d", unnamed_scope_counter++);
                rapidjson::Value new_val;
                new_val.SetString(tmp, strlen(tmp));
                iter->value.AddMember("scope", new_val, json_parser.GetAllocator());
            }

            importNamedParameter(iter->name.GetString(), std::string(iter->value["solver"].GetString()), current_scope, std::string(iter->value["scope"].GetString()));
            import_json_object(iter->value, false);
        }
        else if (iter->value.IsInt())
        {
            //printf("Parsing as int\n");
            importNamedParameter(iter->name.GetString(), iter->value.GetInt(), current_scope, default_new_scope);
        }
        else if (iter->value.IsDouble())
        {
            //printf("Parsing as double\n");
            importNamedParameter(iter->name.GetString(), iter->value.GetDouble(), current_scope, default_new_scope);
        }
        else if (iter->value.IsString())
        {
            //printf("Parsing as string\n");
            importNamedParameter(iter->name.GetString(), std::string(iter->value.GetString()), current_scope, default_new_scope);
        }
        else
        {
            std::string err = "Cannot import parameter \"" + std::string(iter->name.GetString()) + "\" of type " + std::string(json_type_names[iter->value.GetType()]);
        }
    }
}

AMGX_ERROR AMG_Config::parse_json_file(const char *filename)
{
    std::ifstream fin;

    try
    {
        // Store the file content into a string
        std::string params = "";
        fin.open(filename);

        if (!fin)
        {
            char error[500];
            sprintf(error, "Error opening file '%s'", filename);
            FatalError(error, AMGX_ERR_IO);
        }

        while (!fin.eof())
        {
            std::string line;
            std::getline(fin, line);

            //line=trim(line);
            if (line.empty())
            {
                continue;
            }

            params += (line + "\n");
        }

        fin.close();

        // start parsing
        if (json_parser.Parse<0>(params.c_str()).HasParseError())
        {
            std::string tmp = "Cannot read file as JSON object, trying as AMGX config\n";
            amgx_distributed_output(tmp.c_str(), tmp.length());
            return AMGX_ERR_NOT_IMPLEMENTED; //
        }

        // write json cfg to stdout
        /*rapidjson::FileStream f(stdout);
        rapidjson::PrettyWriter<rapidjson::FileStream> writer(f);
        json_parser.Accept(writer);
        std::cout << std::endl;*/
        import_json_object(json_parser, true);
    }
    catch (amgx_exception &e)
    {
        amgx_distributed_output(e.what(), strlen(e.what()));

        if (fin) { fin.close(); }

        return e.reason();
    }
    catch (...)
    {
        return AMGX_ERR_UNKNOWN;
    }

    return AMGX_OK;
}

AMGX_ERROR AMG_Config::parse_json_string(const char *str)
{
    try
    {
        // start parsing
        if (json_parser.Parse<0>(str).HasParseError())
        {
            std::string tmp = "Cannot read file as JSON object, trying as AMGX config\n";
            amgx_distributed_output(tmp.c_str(), tmp.length());
            return AMGX_ERR_NOT_IMPLEMENTED; //
        }

        /*rapidjson::FileStream f(stdout);
        rapidjson::PrettyWriter<rapidjson::FileStream> writer(f);
        json_parser.Accept(writer);
        std::cout << std::endl;
        */
        import_json_object(json_parser, true);
    }
    catch (amgx_exception &e)
    {
        amgx_distributed_output(e.what(), strlen(e.what()));
        return e.reason();
    }
    catch (...)
    {
        return AMGX_ERR_UNKNOWN;
    }

    return AMGX_OK;
}

void getParameterValueString(std::string &buffer, const std::type_info *type, const Parameter &param)
{
    if (*type == typeid(double))
    {
        buffer.resize(32);
#ifdef _WIN32
        _snprintf(
#else
        snprintf(
#endif
            &buffer[0], 31, "%f", param.get<double>());
    }
    else if (*type == typeid(size_t))
    {
        buffer.resize(32);
#ifdef _WIN32
        _snprintf(
#else
        snprintf(
#endif
            &buffer[0], 31, "%zu", param.get<size_t>());
    }
    else if (*type == typeid(int))
    {
        buffer.resize(32);
#ifdef _WIN32
        _snprintf(
#else
        snprintf(
#endif
            &buffer[0], 31, "%d", param.get<int>());
    }
    else if (*type == typeid(std::string))
    {
        buffer = param.get<std::string>();
    }
    else if (*type == typeid(AlgorithmType))
    {
        buffer.assign(getString(param.get<AlgorithmType>()));
    }
    else if (*type == typeid(ViewType))
    {
        buffer.assign(getString(param.get<ViewType>()));
    }
    else if (*type == typeid(ColoringType))
    {
        buffer.assign(getString(param.get<ColoringType>()));
    }
    else if (*type == typeid(BlockFormat))
    {
        buffer.assign(getString(param.get<BlockFormat>()));
    }
    else if (*type == typeid(NormType))
    {
        buffer.assign(getString(param.get<NormType>()));
    }
    else
    {
        FatalError("Unknown type met while processing parameter", AMGX_ERR_CONFIGURATION)
    }
}

// Not templating this one because we want to separate strings, doubles and ints in JSON document instead of writing everything as strings
void fillJSONValueWithParameter(rapidjson::Value &val, const ParameterDescription &desc, rapidjson::Document::AllocatorType &allocator)
{
    std::string buffer;
    rapidjson::Value default_value;
    {
        if (*(desc.type) == typeid(int))
        {
            default_value.SetInt(desc.default_value.get<int>());
        }
        else if (*(desc.type) == typeid(size_t))
        {
            default_value.SetInt(desc.default_value.get<size_t>());
        }
        else if (*(desc.type) == typeid(double))
        {
            default_value.SetInt(desc.default_value.get<double>());
        }
        else
        {
            getParameterValueString(buffer, desc.type, desc.default_value);
            default_value.SetString(buffer.c_str(), allocator);
        }
    }
    val.AddMember("default_value", default_value, allocator);

    if (desc.allowed_values.pm_type != PM_NOT_SET)
    {
        if (desc.allowed_values.pm_type == PM_SET)
        {
            rapidjson::Value allowed_values_obj(rapidjson::kArrayType);
            const std::vector<Parameter> &values_set = desc.allowed_values.value_set;
            std::string buffer;

            for (int i = 0; i < values_set.size(); i++)
            {
                if (*(desc.type) == typeid(int))
                {
                    allowed_values_obj.PushBack(values_set[i].get<int>(), allocator);
                }
                else if (*(desc.type) == typeid(size_t))
                {
                    allowed_values_obj.PushBack(values_set[i].get<size_t>(), allocator);
                }
                else if (*(desc.type) == typeid(double))
                {
                    allowed_values_obj.PushBack(values_set[i].get<double>(), allocator);
                }
                else
                {
                    getParameterValueString(buffer, desc.type, values_set[i]);
                    rapidjson::Value temp_value(buffer.c_str(), allocator);
                    allowed_values_obj.PushBack(temp_value, allocator);
                }
            }

            val.AddMember("allowed_values", allowed_values_obj, allocator);
        }
        else if (desc.allowed_values.pm_type == PM_MINMAX)
        {
            rapidjson::Value allowed_values_obj(rapidjson::kObjectType);
            const std::pair<Parameter, Parameter> &values_pair = desc.allowed_values.value_min_max;
            std::string buffer;
            {
                if (*(desc.type) == typeid(int))
                {
                    allowed_values_obj.AddMember("min", values_pair.first.get<int>(), allocator);
                    allowed_values_obj.AddMember("max", values_pair.second.get<int>(), allocator);
                }
                else if (*(desc.type) == typeid(size_t))
                {
                    allowed_values_obj.AddMember("min", values_pair.first.get<size_t>(), allocator);
                    allowed_values_obj.AddMember("max", values_pair.second.get<size_t>(), allocator);
                }
                else if (*(desc.type) == typeid(double))
                {
                    allowed_values_obj.AddMember("min", (double)(values_pair.first.get<double>()), allocator);
                    allowed_values_obj.AddMember("max", (double)(values_pair.second.get<double>()), allocator);
                }
                else
                {
                    getParameterValueString(buffer, desc.type, values_pair.first);
                    rapidjson::Value temp_value(buffer.c_str(), allocator);
                    allowed_values_obj.AddMember("min", temp_value, allocator);
                    getParameterValueString(buffer, desc.type, values_pair.second);
                    temp_value.SetString(buffer.c_str(), allocator);
                    allowed_values_obj.AddMember("max", temp_value, allocator);
                }
            }
            val.AddMember("allowed_values", allowed_values_obj, allocator);
        }
    }
}

#endif

AMGX_ERROR AMG_Config::write_parameters_description_json(const char *filename)
{
    try
    {
#ifndef RAPIDJSON_DEFINED
        FatalError("This build does not support JSON.", AMGX_ERR_NOT_IMPLEMENTED);
#else
        rapidjson::Document json_out;
        json_out.SetObject();

        for (ParamDesc::const_iterator iter = param_desc.begin(); iter != param_desc.end(); iter++)
        {
            // new entry
            rapidjson::Value param_object(rapidjson::kObjectType);
            // adding description
            param_object.AddMember("description", iter->second.description.c_str(), json_out.GetAllocator());
            // adding parameter type name
            rapidjson::Value param_type(AMG_Config::getParamTypeName(iter->second.type).c_str(), json_out.GetAllocator());
            param_object.AddMember("parameter_type", param_type, json_out.GetAllocator());
            //adding default value
            fillJSONValueWithParameter(param_object, iter->second, json_out.GetAllocator());
            // adding entry to output doc
            json_out.AddMember(iter->first.c_str(), param_object, json_out.GetAllocator());
        }

        // write json cfg to stdout
        FILE *fout = fopen(filename, "w");

        if (!fout)
        {
            std::string tmp = "Cannot open output file" + std::string(filename) + " to write parameters description";
            FatalError(tmp.c_str(), AMGX_ERR_IO);
        }

        rapidjson::FileStream f(fout);
        rapidjson::PrettyWriter<rapidjson::FileStream> writer(f);
        json_out.Accept(writer);
        fclose(fout);
#endif
    }
    catch (amgx_exception &e)
    {
        amgx_output(e.what(), strlen(e.what()));
        return e.reason();
    }
    catch (...)
    {
        return AMGX_ERR_UNKNOWN;
    }

    return AMGX_OK;
}


AMGX_ERROR AMG_Config::parseFile(const char *filename)
{
    try
    {
#ifdef RAPIDJSON_DEFINED
        AMGX_ERROR json_ret = parse_json_file(filename);

        // if parsed ok, then we are done here, otherwise - try to parse as a string
        if (json_ret == AMGX_OK)
        {
            return json_ret;
        }

#endif
        std::string params;

        // Get the parameter string corresponding to file
        if (getParameterStringFromFile(filename, params) != AMGX_OK)
        {
            std::string err = "Error parsing parameter file: " + std::string(filename);
            FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
        }

        // Read the config version
        int config_version = getConfigVersion(params);

        // Parse the string
        if (parseString(params, config_version) != AMGX_OK)
        {
            std::string err = "Error parsing parameter string obtained from file: " + params;
            FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
        }
    }
    catch (amgx_exception &e)
    {
        amgx_output(e.what(), strlen(e.what()));
        return e.reason();
    }
    catch (...)
    {
        return AMGX_ERR_UNKNOWN;
    }

    return AMGX_OK;
}

template <typename Type>
void AMG_Config::getParameter(const std::string &name, Type &value, const std::string &current_scope, std::string &new_scope) const
{
    //verify the parameter has been registered
    ParamDesc::const_iterator desc_iter = param_desc.find(name);
    std::string err;

    if (desc_iter == param_desc.end())
    {
        err = "getParameter error: '" + std::string(name) + "' not found\n";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }

    //verify the types match
    if (desc_iter->second.type != &typeid(Type))
    {
        err = "getParameter error: '" + std::string(name) + "' type miss match\n";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }

    // Check if the parameter name/scope pair has been set
    ParamDB::const_iterator param_iter = m_params.find(make_pair(current_scope, name));

    // Get the value and new_scope
    if (param_iter == m_params.end())
    {
        value = desc_iter->second.default_value.get<Type>();
        new_scope = "default";
    }
    else
    {
        value = param_iter->second.second.get<Type>();     //return the parameter value
        new_scope = param_iter->second.first;              //return the new_scope associated with parameter
    }
}

template <typename Type>
Type AMG_Config::getParameter(const std::string &name, const std::string &current_scope) const
{
    Type value;
    std::string new_scope;
    // For cases where the new scope is not required
    getParameter(name, value, current_scope, new_scope);
    return value;
}



template <typename Type>
void AMG_Config::setParameter(std::string name, Type value, const std::string &current_scope, const std::string &new_scope)
{
    //verify that the parameter has been registered
    ParamDesc::iterator iter = param_desc.find(name);
    std::string err;

    if (iter == param_desc.end())
    {
        err = "setParameter error: '" + std::string(name) + "' not found\n";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }

    if (iter->second.type != &typeid(Type))
    {
        err = "setParameter error: '" + std::string(name) + "' type miss match\n";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }

    m_params[make_pair(current_scope, name)] = make_pair(new_scope, value);
}


template <typename Type>
void AMG_Config::setParameter(std::string name, Type value, const std::string &current_scope)
{
    setParameter(name, value, current_scope, "default");
}



/*
template <typename Type>
void AMG_Config::setParameter(std::string name, Type value) {
  setParameter(name,value,"default","default");
}
*/

template <>
void AMG_Config::setParameter(std::string name, void *value, const std::string &current_scope, const std::string &new_scope)
{
    //verify that the parameter has been registered
    ParamDesc::iterator iter = param_desc.find(name);
    std::string err;

    if (iter == param_desc.end())
    {
        err = "setParameter error: '" + std::string(name) + "' not found\n";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }

    if (iter->second.type != &typeid(void *))
    {
        err = "setParameter error: '" + std::string(name) + "' type miss match\n";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }

    m_params[make_pair(current_scope, name)] = make_pair(new_scope, value);
}

template <>
void AMG_Config::setParameter(std::string name, void *value, const std::string &current_scope)
{
    setParameter<void *>(name, value, current_scope, "default");
}

std::string AMG_Config::getParameterString(Parameter &parameter, ParameterDescription &param_desc)
{
    std::stringstream ss;

    if (*(param_desc.type) == typeid(std::string))
    {
        ss << parameter.get<std::string>() ;
    }
    else if (*(param_desc.type) == typeid(int))
    {
        ss << parameter.get<int>() ;
    }
    else if (*(param_desc.type) == typeid(size_t))
    {
        ss << parameter.get<size_t>() ;
    }
    else if (*(param_desc.type) == typeid(float))
    {
        ss << parameter.get<float>() ;
    }
    else if (*(param_desc.type) == typeid(double))
    {
        ss << parameter.get<double>() ;
    }
    else if (*(param_desc.type) == typeid(AlgorithmType))
    {
        ss << getString(parameter.get<AlgorithmType>()) ;
    }
    else if (*(param_desc.type) == typeid(ViewType))
    {
        ss << getString(parameter.get<ViewType>()) ;
    }
    else if (*(param_desc.type) == typeid(ColoringType))
    {
        ss << getString(parameter.get<ColoringType>()) ;
    }
    else if (*(param_desc.type) == typeid(BlockFormat))
    {
        ss << getString(parameter.get<BlockFormat>()) ;
    }
    else if (*(param_desc.type) == typeid(NormType))
    {
        ss << getString(parameter.get<NormType>()) ;
    }
    else if (*(param_desc.type) == typeid(void *))
    {
        ss << parameter.get<void *>();
    }
    else
    {
        std::string err = "getParameterString is not implemented for the datatype of value'" + param_desc.name + "'";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }

    return ss.str();
}


void AMG_Config::printOptions()
{
    std::stringstream ss;

    for (ParamDesc::iterator iter = param_desc.begin(); iter != param_desc.end(); iter++)
    {
        ss << "           " << iter->second.name << ": " << iter->second.description << std::endl;
    }

    amgx_output(ss.str().c_str(), ss.str().length());
}

void AMG_Config::printAMGConfig()
{
    std::stringstream config_ss;
    std::stringstream ss;
    int devId;
    cudaGetDevice(&devId);
    cudaDeviceProp deviceProp = getDeviceProperties();
    //ss << "HP Scalar Type: " << scalar_hp << std::endl;
    //ss << "LP Scalar Type: " << scalar_lp << std::endl;
    ss << "Device " << devId << ": " << deviceProp.name << std::endl;
    ss << "AMG Configuration: "  << std::endl;
    config_ss << std::endl;
    config_ss << "Default values:" << std::endl ;
    config_ss << std::endl;

    for (ParamDesc::iterator iter = param_desc.begin(); iter != param_desc.end(); iter++)
    {
        config_ss << "            " << iter->second.name << " = ";
        config_ss << getParameterString(iter->second.default_value, iter->second);
        config_ss << std::endl;
    }

    config_ss << std::endl;
    config_ss << " User-defined parameters:" << std::endl;
    config_ss << " Current_scope:parameter_name(new_scope) = parameter_value" << std::endl;
    config_ss << std::endl;

    for (ParamDB::iterator iter = m_params.begin(); iter != m_params.end(); iter++)
    {
        // Search for the name in ParamDesc database
        ParamDesc::iterator desc_iter = param_desc.find(iter->first.second);
        config_ss << "            " ;
        //if (iter->first.first != "default")
        config_ss << iter->first.first << ":";
        config_ss <<  iter->first.second;

        if ( iter->second.first != "default")
        {
            config_ss << "(" << iter->second.first << ")";
        }

        config_ss << " = " ;
        config_ss << getParameterString(iter->second.second, desc_iter->second);
        config_ss << std::endl;
    }

    config_ss << std::endl;
    amgx_output(ss.str().c_str(), ss.str().length());
    amgx_output(config_ss.str().c_str(), config_ss.str().length());
}

AMGX_ERROR AMG_Config::checkString(std::string &str)
{
    std::string::iterator it;

    for (it = str.begin(); it < str.end(); it++)
    {
        if (!allowed_symbol(*it))
        {
            return AMGX_ERR_CONFIGURATION;
        }
    }

    // check if string is empty

    if (str.find_first_not_of(' ') == std::string::npos)
    {
        return AMGX_ERR_CONFIGURATION;
    }

    return AMGX_OK;
}

void AMG_Config::extractParamInfo(const std::string &str, std::string &name, std::string &value, std::string &current_scope, std::string &new_scope)
{
    std::string tmp(str);

    //locate the split
    if ( std::count(tmp.begin(), tmp.end(), '=') != 1)
    {
        tmp = "Incorrect config entry (number of equal signs is not 1) : " + str;
        FatalError(tmp.c_str(), AMGX_ERR_CONFIGURATION);
    }

    size_t split_loc = tmp.find("=");
    value = tmp.substr(split_loc + 1);
    name = tmp.substr(0, split_loc);
    // Extract the new scope
    int num_left_brackets = std::count(name.begin(), name.end(), '(');

    if ( num_left_brackets == std::count(name.begin(), name.end(), ')') && (num_left_brackets == 0 || num_left_brackets == 1) )
    {
        if (num_left_brackets == 0)
        {
            new_scope = "default";
        }
        else if (num_left_brackets == 1)
        {
            size_t split_loc_l = name.find("(");
            size_t split_loc_r = name.find(")");
            new_scope = name.substr(split_loc_l + 1, split_loc_r - split_loc_l - 1);
            name = name.substr(0, split_loc_l);

            if (checkString(trim(new_scope)) != AMGX_OK)
            {
                std::string tmp = "Incorrect config entry (invalid symbol or empty string after trimming new_scope): " + str;
                FatalError(tmp.c_str(), AMGX_ERR_CONFIGURATION);
            }

            if (new_scope == "default")
            {
                tmp = "Incorrect config entry (new scope cannot be default scope): " + str;
                FatalError(tmp.c_str(), AMGX_ERR_CONFIGURATION);
            }
        }
    }
    else
    {
        tmp = "Incorrect config entry (incorrect number of parentheses or unbalanced parantheses): " + str;
        FatalError(tmp.c_str(), AMGX_ERR_CONFIGURATION);
    }

    // Extract current scope and name
    split_loc = name.find(":");
    int num_colon = std::count(name.begin(), name.end(), ':');

    if (num_colon == 0)
    {
        // do nothing, will check later if name==solver
        current_scope = "default";
    }
    else if (num_colon == 1)
    {
        size_t split_loc_l = name.find(":");
        current_scope = name.substr(0, split_loc_l);
        name = name.substr(split_loc_l + 1);

        if (checkString(trim(current_scope)) != AMGX_OK)
        {
            std::string tmp = "Incorrect config entry (invalid string or empty string after trimming current_scope): " + str;
            FatalError(tmp.c_str(), AMGX_ERR_CONFIGURATION);
        }
    }
    else
    {
        tmp = "Incorrect config entry (number of colons is > 1): " + str;
        FatalError(tmp.c_str(), AMGX_ERR_CONFIGURATION);
    }

    // -----------------
    // strip strings
    // -----------------
    if (checkString(trim(value)) != AMGX_OK || checkString(trim(name)) != AMGX_OK)
    {
        std::string tmp = "Incorrect config entry (invalid string or empty string after stripping name or value): " + str;
        FatalError(tmp.c_str(), AMGX_ERR_CONFIGURATION);
    }
}


void AMG_Config::setParameter(const std::string &str)
{
    std::string name, value, current_scope, new_scope;
    std::string tmp;
    extractParamInfo(str, name, value, current_scope, new_scope);

    // Add new_scope to scope vector
    if ( find(m_scope_vector.begin(), m_scope_vector.end(), new_scope) == m_scope_vector.end())
    {
        m_scope_vector.push_back(new_scope);
    }
    else if (new_scope != "default" && !getAllowConfigurationMod())
    {
        tmp = "Incorrect config entry (new scope already defined): " + str;
        FatalError(tmp.c_str(), AMGX_ERR_CONFIGURATION);
    }

    // extract the name, value, new_scope and old_scope
    //verify parameter was registered
    ParamDesc::iterator iter = param_desc.find(std::string(name));

    if (iter == param_desc.end())
    {
        std::string err = "Variable '" + std::string(name) + "' not registered";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }

    if ( (name == "determinism_flag" || name == "block_format" || name == "separation_interior" || name == "separation_exterior" || name == "min_rows_latency_hiding" || name == "fine_level_consolidation" || name == "use_cuda_ipc_consolidation") && current_scope != "default" )
    {
        std::string err = "Incorrect config entry. Parameter " + name + " can only be specified with default scope.";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }

    // Check that new scope is only associated with a solver
    if (new_scope != "default" && find(m_solver_list.begin(), m_solver_list.end(), name) == m_solver_list.end() )
    {
        std::string err = "Incorrect config entry. New scope can only be associated with a solver. new_scope=" + new_scope + ", name=" + name + ".";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }

    // Set the new parameter name, value
    if (*(iter->second.type) == typeid(std::string))
    {
        setParameter(name, value, current_scope, new_scope);
    }
    else if (*(iter->second.type) == typeid(int))
    {
        setParameter(name, getValue<int>(value.c_str()), current_scope, new_scope);
    }
    else if (*(iter->second.type) == typeid(size_t))
    {
        setParameter(name, getValue<size_t>(value.c_str()), current_scope, new_scope);
    }
    else if (*(iter->second.type) == typeid(float))
    {
        setParameter(name, getValue<float>(value.c_str()), current_scope, new_scope);
    }
    else if (*(iter->second.type) == typeid(double))
    {
        setParameter(name, getValue<double>(value.c_str()), current_scope, new_scope);
    }
    else if (*(iter->second.type) == typeid(AlgorithmType))
    {
        setParameter(name, getValue<AlgorithmType>(value.c_str()), current_scope, new_scope);
    }
    else if (*(iter->second.type) == typeid(ViewType))
    {
        setParameter(name, getValue<ViewType>(value.c_str()), current_scope, new_scope);
    }
    else if (*(iter->second.type) == typeid(ColoringType))
    {
        setParameter(name, getValue<ColoringType>(value.c_str()), current_scope, new_scope);
    }
    else if (*(iter->second.type) == typeid(BlockFormat))
    {
        setParameter(name, getValue<BlockFormat>(value.c_str()), current_scope, new_scope);
    }
    else if (*(iter->second.type) == typeid(NormType))
    {
        setParameter(name, getValue<NormType>(value.c_str()), current_scope, new_scope);
    }
    else
    {
        std::string err = "getValue is not implemented for the datatype of variable '" + name + "'";
        FatalError(err.c_str(), AMGX_ERR_CONFIGURATION);
    }
}

AMG_Config::AMG_Config() : ref_count(1), m_latest_config_version(2), m_config_version(0), m_allow_cfg_mod(0)
{
    m_scope_vector.push_back("default");
    m_solver_list.push_back("solver");
    m_solver_list.push_back("preconditioner");
    m_solver_list.push_back("smoother");
    m_solver_list.push_back("coarse_solver");
    m_solver_list.push_back("cpr_first_stage_preconditioner");
    m_solver_list.push_back("cpr_second_stage_preconditioner");
}

void AMG_Config::clear()
{
    m_params.clear();
    m_scope_vector.clear();
    m_scope_vector.push_back("default");
}


// Template specialization
template std::string AMG_Config::getParameter(const std::string &, const std::string &) const;
template AlgorithmType AMG_Config::getParameter(const std::string &, const std::string &) const;
template ViewType AMG_Config::getParameter(const std::string &, const std::string &) const;
template ColoringType AMG_Config::getParameter(const std::string &, const std::string &) const;
template BlockFormat AMG_Config::getParameter(const std::string &, const std::string &) const;
template NormType AMG_Config::getParameter(const std::string &, const std::string &) const;
template int AMG_Config::getParameter(const std::string &, const std::string &) const;
template size_t AMG_Config::getParameter(const std::string &, const std::string &) const;
template float AMG_Config::getParameter(const std::string &, const std::string &) const;
template double AMG_Config::getParameter(const std::string &, const std::string &) const;
template void *AMG_Config::getParameter(const std::string &, const std::string &) const;

template void AMG_Config::getParameter(const std::string &, std::string &, const std::string &, std::string &) const;
template void AMG_Config::getParameter(const std::string &, ViewType &, const std::string &, std::string &) const;
template void AMG_Config::getParameter(const std::string &, ColoringType &, const std::string &, std::string &) const;
template void AMG_Config::getParameter(const std::string &, AlgorithmType &, const std::string &, std::string &) const;
template void AMG_Config::getParameter(const std::string &, BlockFormat &, const std::string &, std::string &) const;
template void AMG_Config::getParameter(const std::string &, NormType &, const std::string &, std::string &) const;
template void AMG_Config::getParameter(const std::string &, int &, const std::string &, std::string &) const;
template void AMG_Config::getParameter(const std::string &, size_t &, const std::string &, std::string &) const;
template void AMG_Config::getParameter(const std::string &, float &, const std::string &, std::string &) const;
template void AMG_Config::getParameter(const std::string &, double &, const std::string &, std::string &) const;
template void AMG_Config::getParameter(const std::string &, void *&, const std::string &, std::string &) const;

template void AMG_Config::setParameter(std::string, std::string, const std::string &) ;
template void AMG_Config::setParameter(std::string, AlgorithmType, const std::string &) ;
template void AMG_Config::setParameter(std::string, ViewType, const std::string &) ;
template void AMG_Config::setParameter(std::string, ColoringType, const std::string &) ;
template void AMG_Config::setParameter(std::string, BlockFormat, const std::string &) ;
template void AMG_Config::setParameter(std::string, NormType, const std::string &) ;
template void AMG_Config::setParameter(std::string, int, const std::string &) ;
template void AMG_Config::setParameter(std::string, size_t, const std::string &) ;
template void AMG_Config::setParameter(std::string, float, const std::string &) ;
template void AMG_Config::setParameter(std::string, double, const std::string &) ;

template void AMG_Config::setParameter(std::string, std::string, const std::string &, const std::string &) ;
template void AMG_Config::setParameter(std::string, AlgorithmType, const std::string &, const std::string &) ;
template void AMG_Config::setParameter(std::string, ViewType, const std::string &, const std::string &) ;
template void AMG_Config::setParameter(std::string, ColoringType, const std::string &, const std::string &) ;
template void AMG_Config::setParameter(std::string, BlockFormat, const std::string &, const std::string &) ;
template void AMG_Config::setParameter(std::string, NormType, const std::string &, const std::string & ) ;
template void AMG_Config::setParameter(std::string, int, const std::string &, const std::string &) ;
template void AMG_Config::setParameter(std::string, size_t, const std::string &, const std::string &) ;
template void AMG_Config::setParameter(std::string, float, const std::string &, const std::string &) ;
template void AMG_Config::setParameter(std::string, double, const std::string &, const std::string &) ;


AMG_Configuration::AMG_Configuration()
{
    amg_config = new AMG_Config;
};

AMG_Configuration::AMG_Configuration(const AMG_Configuration &cfg)
{
    amg_config = cfg.amg_config;
    amg_config->ref_count++;
}

AMG_Configuration &AMG_Configuration::operator=(const AMG_Configuration &cfg)
{
    amg_config = cfg.amg_config;
    amg_config->ref_count++;
    return *this;
}

AMG_Configuration::~AMG_Configuration()
{
    if (--amg_config->ref_count == 0)
    {
        delete amg_config;
    }
};

/********************************************
 * Gets a parameter from the database and
 * throws an exception if it does not exist.
 *********************************************/
template <typename Type> Type AMG_Configuration::getParameter(std::string name) const {amg_config->getParameter<Type>(name, "default");}

/**********************************************
 * Sets a parameter in the database
 * throws an exception if it does not exist.
 *********************************************/
template <typename Type> void AMG_Configuration::setParameter(std::string name, Type value, const std::string &current_scope) {amg_config->setParameter(name, value, current_scope);}
template <> void AMG_Configuration::setParameter(std::string name, int value, const std::string &current_scope) {amg_config->setParameter(name, value, current_scope);}
template <> void AMG_Configuration::setParameter(std::string name, double value, const std::string &current_scope) {amg_config->setParameter(name, value, current_scope);}
template <> void AMG_Configuration::setParameter(std::string name, void *value, const std::string &current_scope) {amg_config->setParameter(name, value, current_scope);}

/****************************************************
 * Parse paramters string
 * scope:name(new_scope)=value, scope:name(new_scope)=value, ..., scope:name(new_scope)=value
 * and store the variables in the parameter database
 ****************************************************/
AMGX_ERROR AMG_Configuration::parseParameterString(const char *str) {return amg_config->parseParameterString(str);}

/****************************************************
* Parse a config file  in the format
* scope:name(new_scope)=value
* scope:name(new_scope)=value
* ...
* scope:name(new_scope)=value
* and store the variables in the parameter database
****************************************************/

AMGX_ERROR AMG_Configuration::parseFile(const char *filename) {return amg_config->parseFile(filename); }

/****************************************************
 * Parse paramters string
 * scope:name(new_scope)=value, scope:name(new_scope)=value, ..., scope:name(new_scope)=value
 * and store the variables in the parameter database
 ****************************************************/
AMGX_ERROR AMG_Configuration::parseParameterStringAndFile(const char *str, const char *filename) {return amg_config->parseParameterStringAndFile(str, filename);}

/****************************************************
 * Print the options for AMG
 ***************************************************/
void AMG_Configuration::printOptions() {AMG_Config::printOptions(); }

} // namespace amgx
