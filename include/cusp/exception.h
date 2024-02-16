// SPDX-FileCopyrightText: 2008 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

/*! \file exception.h
 *  \brief Cusp exceptions
 */

#pragma once

#include <cusp/detail/config.h>

#include <string>
#include <stdexcept>

namespace cusp
{

    class exception : public std::exception
    {
        public:
            exception(const exception& exception_) : message(exception_.message) {}
            exception(const std::string& message_) : message(message_) {}
            ~exception() throw() {}
            const char* what() const throw() { return message.c_str(); }

        protected:
            std::string message;
    };
    
    class not_implemented_exception : public exception
    {
        public:
            template <typename MessageType>
            not_implemented_exception(const MessageType& message) : exception(message) {}
    };

    class io_exception : public exception
    {
        public:
            template <typename MessageType>
            io_exception(const MessageType& message) : exception(message) {}
    };

    class invalid_input_exception : public exception
    {
        public:
            template <typename MessageType>
            invalid_input_exception(const MessageType& message) : exception(message) {}
    };
    
    class format_exception : public exception
    {
        public:
            template <typename MessageType>
            format_exception(const MessageType& message) : exception(message) {}
    };

    class format_conversion_exception : public format_exception
    {
        public:
            template <typename MessageType>
            format_conversion_exception(const MessageType& message) : format_exception(message) {}
    };
    
    class runtime_exception : public exception
    {
        public:
            template <typename MessageType>
            runtime_exception(const MessageType& message) : exception(message) {}
    };

} // end namespace cusp

