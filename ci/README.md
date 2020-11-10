Continuous integration
===

**WIP**: Adding continuous integration to AmgX is currently a work in progress.

* [`./ci/run.sh`](run.sh) runs the whole CI pipeline locally: it builds the
  docker containers for each supported environment, builds AmgX for that
  environment, and runs the AmgX tests. 
* [`./ci/test.sh`](test.sh) performs a clean run of the AmgX tests.

The containers are specified using [`HPCCM`], see [`containers/`](containers).

[`HPCCM`]: https://github.com/NVIDIA/hpc-container-maker

The behavior of the CI system is configured using the following environment variables:

* `AMGX_CI_CONTAINERS="<list>"`: list of containers to test. By default all
  containers are tested.
* `AMGX_CI_KEEP_BUILD=0|1`: whether the build directories are preserved across
  CI runs. The default is `0`, i.e., the build directories are cleaned and amgx
  is re-built from scratch on every run.
  
* `AMGX_CI_CONTAINER_FILE`: dump container build recipe to a file in the current
  working directory: `Dockerfile_${baseimage}`.

For example, to only test the `x86_64-ubuntu18.04-gnu7-cuda10.2.py` container,
preserving the build directory (e.g. during development):

```shell
AMGX_CI_CONTAINERS="x86_64-ubuntu18.04-gnu7-cuda10.2.py" AMGX_CI_KEEP_BUILD=1 ./ci/run.sh
```
