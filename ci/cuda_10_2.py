"""
AmgX base image for CUDA 10.2
"""

Stage0 += comment(__doc__, reformat=False)
Stage0 += baseimage(image='nvidia/cuda:10.2-devel-ubuntu18.04')

# Last compiler supported for Ubuntu 18.04 by CUDA 10.2
# https://docs.nvidia.com/cuda/archive/10.2/cuda-installation-guide-linux/index.html#system-requirements
compiler = gnu()
Stage0 += compiler

# Current minimum version required by AMGX
Stage0 += cmake(eula=True, version='2.8.10')

# MPI
Stage0 += mlnx_ofed(version='5.0-2.1.8.0')
Stage0 += gdrcopy(ldconfig=True, version='2.0')
Stage0 += knem(ldconfig=True, version='1.1.3')
Stage0 += ucx(gdrcopy=True, knem=True, ofed=True, cuda=True)
Stage0 += openmpi(
    cuda=True,
    infiniband=True,
    version='4.0.3',
    pmix=True,
    ucx=True,
    toolchain=compiler.toolchain
)
Stage0 += environment(multinode_vars = {
    'OMPI_MCA_pml': 'ucx',
    'OMPI_MCA_btl': '^smcuda,vader,tcp,uct,openib',
    'UCX_MEMTYPE_CACHE': 'n',
    'UCX_TLS': 'rc,cuda_copy,cuda_ipc,gdr_copy,sm'
})
