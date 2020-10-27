"""
AmgX base image: x86_64-ubuntu20.04-nvhpc20.9
"""
import posixpath
Stage0 += comment(__doc__, reformat=False)
Stage0 += baseimage(image='ubuntu:20.04')

compiler = nvhpc(eula=True, version='20.9', cuda_multi=False, cuda='11.0')
# WAR: nvhpc should be doing this
compiler.toolchain.CUDA_HOME = '/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/cuda/11.0'
Stage0 += compiler

# Current minimum version required by AMGX
Stage0 += cmake(eula=True, version='3.7.0')

# MPI
Stage0 += mlnx_ofed(version='5.0-2.1.8.0')

Stage0 += gdrcopy(ldconfig=True, version='2.0')
Stage0 += knem(ldconfig=True, version='1.1.3')

# BUG: this should just work
# Stage0 += ucx(gdrcopy=True,knem=True,ofed=True,cuda=True)
Stage0 += ucx(
    # WAR: should not be necessary:
    build_environment={
        'LD_LIBRARY_PATH': '/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}',
    },
    gdrcopy=True,
    knem=True,
    ofed=True,
    cuda=compiler.toolchain.CUDA_HOME,
    # WAR: should not be necessary, required because OpenMPI cannot find UCX at
    # default install location
    prefix='/usr/local/ucx'
)

Stage0 += openmpi(
    cuda=True,
    infiniband=True,
    version='4.0.3',
    pmix=True,
    # WAR: should not be necessary: ucx=True should do the right thing
    ucx='/usr/local/ucx',
    toolchain=compiler.toolchain
)
Stage0 += environment(multinode_vars = {
    'OMPI_MCA_pml': 'ucx',
    'OMPI_MCA_btl': '^smcuda,vader,tcp,uct,openib',
    'UCX_MEMTYPE_CACHE': 'n',
    'UCX_TLS': 'rc,cuda_copy,cuda_ipc,gdr_copy,sm'
  },
  # WAR: we should have a `compiler.toolchain.environment()` API to do this properly
  variables={
      'CUDA_HOME': compiler.toolchain.CUDA_HOME,
      'CC': compiler.toolchain.CC,
      'CXX': compiler.toolchain.CXX,
      'FC': compiler.toolchain.FC,
      'FC': compiler.toolchain.FC,
      'F90': compiler.toolchain.F90,
      'F77': compiler.toolchain.F77
  }
)
