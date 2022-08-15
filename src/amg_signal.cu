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

#include "amg_signal.h"
#include "misc.h"
#include <signal.h>
#include "stdio.h"

#ifdef _WIN32
#include "windows.h"
#endif

#include "stacktrace.h"

namespace amgx
{

bool SignalHandler::hooked = false;

#ifdef _WIN32

const int NUM_SIGS = 6;
static int SIGNALS[NUM_SIGS] = {SIGINT, SIGILL, SIGABRT, SIGFPE, SIGSEGV, SIGTERM};

/****************************************
 * converts a signal to a string
 ****************************************/
inline const char *getSigString(int sig)
{
    switch (sig)
    {
        case SIGINT:
            return "SIGINT (interrupt)";

        case SIGABRT:
            return "SIGABRT (abort)";

        case SIGFPE:
            return "SIGFPE (floating point exception)";

        case SIGILL:
            return "SIGILL (illegal instruction)";

        case SIGSEGV:
            return "SIGSEGV (segmentation violation)";

        case SIGTERM:
            return "SIGTERM (terminated)";

        default:
            return "UNKNOWN";
    }
}


/*****************************************
 * handles the signals by printing the
 * error message, the stack, and exiting
 * where appropriate
 ****************************************/
inline void handle_signals_win(int sig)
{
    char buf[255];
    _snprintf(buf, 255, "Caught signal %d - %s\n", sig, getSigString(sig));
    amgx_output(buf, strlen(buf));
    exit(1);
}

typedef void (*signal_handler)(int);

signal_handler saved_handlers[NUM_SIGS];

UINT last_error_mode;


void SignalHandler::hook()
{
    last_error_mode = SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);

    for (int i = 0; i < NUM_SIGS; i++)
    {
        saved_handlers[i] = signal(SIGNALS[i], handle_signals_win);
    }

    hooked = true;
}

void SignalHandler::unhook()
{
    if (hooked)
    {
        for (int i = 0; i < NUM_SIGS; i++)
        {
            signal(SIGNALS[i], saved_handlers[i]);
        }

        SetErrorMode(last_error_mode);
    }

    hooked = false;
}

#else

const int NUM_SIGS = 10;
static int SIGNALS[NUM_SIGS] = {SIGINT, SIGQUIT, SIGILL, SIGABRT, SIGFPE, SIGSEGV, SIGTERM, SIGPIPE, SIGUSR1, SIGUSR2};

typedef void (*signal_handler)(int);

static struct sigaction saved_actions[NUM_SIGS];

/****************************************
 * converts a signal to a string
 ****************************************/
inline const char *getSigString(int sig)
{
    switch (sig)
    {
        case SIGINT:
            return "SIGINT (interrupt)";

        case SIGABRT:
            return "SIGABRT (abort)";

        case SIGFPE:
            return "SIGFPE (floating point exception)";

        case SIGILL:
            return "SIGILL (illegal instruction)";

        case SIGSEGV:
            return "SIGSEGV (segmentation violation)";

        case SIGTERM:
            return "SIGTERM (terminated)";

        case SIGQUIT:
            return "SIGQUIT (quit)";

        case SIGPIPE:
            return "SIGPIPE (broken pipe)";

        case SIGUSR1:
            return "SIGUSR1 (user 1)";

        case SIGUSR2:
            return "SIGUSR2 (user 2)";

        default:
            return "UNKNOWN";
    }
}

inline void call_default_sig_handler(int sig)
{
    signal(sig, SIG_DFL);
    raise(sig);
}

/*****************************************
 * handles the signals by printing the
 * error message, the stack, and exiting
 * where appropriate
 ****************************************/
inline void handle_signals(int sig)
{
    char buf[255];
    sprintf(buf, "Caught signal %d - %s\n", sig, getSigString(sig));
    amgx_output(buf, strlen(buf));

    switch (sig)
    {
        case SIGINT:
        case SIGKILL:
        case SIGQUIT:
        case SIGTERM:
            //don't print stack trace since the user interrupted this one
            call_default_sig_handler(sig);
            break;

        case SIGUSR1:
        case SIGUSR2: //user defined signal to print the backtrace but continue running
            printStackTrace();
            break;

        default:
            printStackTrace();
            call_default_sig_handler(sig);
    }
}

void SignalHandler::hook()
{
    struct sigaction action;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    action.sa_handler = handle_signals;

    for (int i = 0; i < NUM_SIGS; i++)
    {
        sigaction(SIGNALS[i], &action, &(saved_actions[i]));
    }

    hooked = true;
}

void SignalHandler::unhook()
{
    if (hooked)
        for (int i = 0; i < NUM_SIGS; i++)
        {
            sigaction(SIGNALS[i], &(saved_actions[i]), NULL);
        }

    hooked = false;
}

#endif

} // namespace amgx

