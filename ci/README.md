Continuous integration
===

**WIP**: Adding continuous integration to AmgX is currently a work in progress.

* [`./ci/run.sh`](run.sh) runs the whole CI pipeline locally: it builds the
  docker containers for each supported environment, builds AmgX for that
  environment, and runs the AmgX tests. It takes as an optional argument a list
  of the environments to test, e.g., `./ci/run.sh "foo.py bar.py"` will run the tests
  for the `foo.py` and `bar.py` containers. If this is not specified, i.e., if only
  `./ci/run.sh` is executed, all environments in the [`containers/`](containers)
  directory are tested.
* [`./ci/test.sh`](test.sh) performs a clean run of the AmgX tests.

The containers are specified using [`HPCCM`], see [`containers/`](containers).

[`HPCCM`]: https://github.com/NVIDIA/hpc-container-maker
