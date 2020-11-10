"""
AmgX base image: x86_64-ubuntu18.04-gnu-cuda11.0
"""

Stage0 += comment(__doc__, reformat=False)
Stage0 += baseimage(image='nvidia/cuda:11.0-devel-ubuntu18.04')

# Last compiler supported for Ubuntu 18.04 by CUDA 11.0
# https://docs.nvidia.com/cuda/archive/11.0/cuda-installation-guide-linux/index.html#system-requirements
compiler = gnu(version='8')
Stage0 += compiler

# Current minimum version required by AMGX
Stage0 += cmake(eula=True, version='3.7.0')

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
