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

#include <amg_config.h>
#include <typeinfo>

const std::string type_demangle(const char *name);

namespace amgx
{

class AuxDB;

class AuxPtrBase
{
        friend class AuxDB;
    protected:
        std::string type_name;
        void force_typename(std::string name) {type_name = name;};
    public:
        virtual void *Get() = 0;
        //virtual void Set(void*) = 0;
        virtual void Free() = 0;

        virtual ~AuxPtrBase()
        {
        };
};

template<class T>
class AuxPtr : public AuxPtrBase
{
        T *ptr;
        bool afthis;
    public:
        AuxPtr(bool af = true): ptr(NULL), afthis(af) { this->type_name = typeid(T).name();};
        AuxPtr(T *_ptr, bool af = true) : ptr(_ptr), afthis(af) { this->type_name = typeid(T).name();};
        virtual ~AuxPtr()
        {
            Free();
        };

        void Free()
        {
            if (afthis)
            {
                delete (T *)(ptr);
                ptr = NULL;
            }
        };

        void *Get()
        {
            return (void *)(ptr);
        }
};


class AuxDB
{

    public:

// has parameter
        bool hasParameter(const std::string &name) const;

// getParameter()
        template <typename Type>
        void getParameter(const std::string &name, Type &value) const
        {
            // Check if the parameter name/scope pair has been set
            ParamDB::const_iterator param_iter = params.find(name);

            // Get the value and new_scope
            if (param_iter != params.end())
            {
                value = param_iter->second.get<Type>();     //return the parameter value
            }
            else
            {
                std::string str("Cannot find data with the name : ");
                str += name;
                FatalError( str.c_str(), AMGX_ERR_BAD_PARAMETERS );
            }
        }

// getParameterPtr
        template <typename Type>
        void getParameterPtr(const std::string &name, Type *&value) const
        {
            // Check if the parameter name/scope pair has been set
            ParamPtrDB::const_iterator ptrparam_iter = ptrparams.find(name);

            if (ptrparam_iter != ptrparams.end())
            {
                if (ptrparam_iter->second->type_name != typeid(Type).name())
                {
                    std::stringstream ss;
                    ss << "Trying to retrieve parameter with name \"" << name << "\" with type " << type_demangle(typeid(Type).name()) << " but it's stored with type " << type_demangle(ptrparam_iter->second->type_name.c_str()) << std::endl;
                    FatalError( ss.str().c_str(), AMGX_ERR_BAD_PARAMETERS );
                }

                value = (Type *)(ptrparam_iter->second->Get());
            }
            else
            {
                std::string str("Cannot find data with the name : ");
                str += name;
                FatalError( str.c_str(), AMGX_ERR_BAD_PARAMETERS );
            }
        }

// setParameterPtr()
        template <typename Type>
        void setParameterPtr (std::string &name, Type *value)
        {
            ParamPtrDB::iterator ptrparam_iter = ptrparams.find(name);

            if (ptrparam_iter != ptrparams.end())
            {
                ptrparams.erase(ptrparam_iter);
            }

            ptrparams[name] = new AuxPtr<Type>(value);
        }

// setParameter()
        template <typename Type>
        void setParameter(std::string &name, Type value)
        {
            Parameter new_val(value);
            params[name] = new_val;
        }

// stuff
        void copyParameters(const AuxDB *src);

        ~AuxDB();
        AuxDB() {};
        AuxDB(const AuxDB &src)
        {
            copyParameters(&src);
        }

        AuxDB &operator= (const AuxDB &src)
        {
            // do the copy
            copyParameters(&src);
            // return the existing object
            return *this;
        }
        void printExistingData() const;

    private:
        typedef std::map< std::string, Parameter >              ParamDB;
        typedef std::map< std::string, AuxPtrBase * >            ParamPtrDB;

        void clearPtrs();

        ParamDB         params;
        ParamPtrDB      ptrparams;
};

class AuxData
{
        AuxDB data;
    public:

        void printExistingData() const { data.printExistingData();}

        void copyAuxData(const AuxData *src)
        {
            data.copyParameters(&src->data);
        }

        template <typename Type>
        void setParameter(std::string &name, Type value)
        {
            data.setParameter<Type>(name, value);
        }

        template <typename Type>
        void setParameter(const char *name, Type value)
        {
            std::string _name(name);
            setParameter<Type>(_name, value);
        }

        template <typename Type>
        void setParameterPtr (std::string &name, Type *value)
        {
            data.setParameterPtr(name, value);
        }

        template <typename Type>
        void setParameterPtr (const char *name, Type *value)
        {
            std::string _name(name);
            setParameterPtr(_name, value);
        }

        template <typename Type>
        void getParameterPtr(const char *name, Type *&value) const
        {
            getParameterPtr<Type>(std::string(name), value);
        }

        template <typename Type>
        Type *getParameterPtr(const char *name) const
        {
            return getParameterPtr<Type>(std::string(name));
        }

        template <typename Type>
        Type *getParameterPtr(const std::string &name) const
        {
            Type *value;
            getParameterPtr(name, value);
            return value;
        }

        template <typename Type>
        void getParameterPtr(const std::string &name, Type *&value) const
        {
            data.getParameterPtr(name, value);
        }

        template <typename Type>
        void getParameter(const std::string &name, Type &value) const
        {
            data.getParameter(name, value);
        }

        template <typename Type>
        Type getParameter(const std::string &name) const
        {
            Type value;
            getParameter(name, value);
            return value;
        }


        template <typename Type>
        void getParameter(const char *name, Type &value) const
        {
            getParameter<Type>(std::string(name), value);
        }

        template <typename Type>
        Type getParameter(const char *name) const
        {
            return getParameter<Type>(std::string(name));
        }

        bool hasParameter(const char *name) const
        {
            return hasParameter(std::string(name));
        }

        bool hasParameter(const std::string &name) const
        {
            return data.hasParameter(name);
        }
};

} // namespace amgx
