// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

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

