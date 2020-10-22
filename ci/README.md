Continuous integration
===

**WIP**: Adding continuous integration to AmgX is currently a work in progress.

* [`./ci/run.sh`](run.sh) runs the whole CI pipeline locally: it builds the
  docker containers for each supported target and runs the AmgX tests.
* [`./ci/test.sh`](test.sh) performs a clean run of the AmgX tests.

The containers are specified using [`HPCCM`], see [`containers/`](containers).

[`HPCCM`]: https://github.com/NVIDIA/hpc-container-maker
