---
name: Compilation issue report
about: Report an issue during compilation of AMGX
title: "[Build]"
labels: build issues
assignees: ''

---

**Describe the issue**

A clear and concise description of what the issue is - this might be configuration error with CMake, or build error.

**Environment information:**
 - OS: [e.g. `Windows 10`, `Ubuntu 22.04`]
 - Compiler version: [e.g. `gcc 9.3.0`, `MSVC 14.31`]
 - CMake version: [e.g. `3.23`]
 - CUDA used for AMGX compilation: [e.g. `CUDA 11.7.0`]
 - MPI version (if applicable): [e.g. `OpenMPI 4.0.3`, `MS-MPI v10.1.2`]
 - AMGX version or commit hash [e.g. `v2.3.0`, `34232979e993d349a03486f7892830a1209b2fc9`]
 - Any related environment variables information

**Configuration information** 

Provide your `cmake` command line that was used for configuration and it's full output.

**Compilation information**

Provide your make command

**Issue information** 

Provide any error messages from your CMake or compiler. It will also greatly help to attach output of `make` command rerun with `VERBOSE=1` to see exact host compiler launch command that issues the error.

**Additional context**

Add any other context about the problem here.
