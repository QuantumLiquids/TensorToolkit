# Install cuTENSOR Without Root

This page covers the two common cluster cases for TensorToolkit GPU builds:

1. The cluster provides cuTENSOR already, often through NVIDIA HPC SDK / NVHPC.
2. The cluster provides CUDA but not cuTENSOR, so you install cuTENSOR under your home directory.

TensorToolkit's CMake now supports both patterns more directly. In many cases,
you only need `-DQLTEN_USE_GPU=ON`; if auto-discovery still fails, set
`CUTENSOR_ROOT`, or set `CUTENSOR_INCLUDE_DIR` and `CUTENSOR_LIBRARY`
explicitly.

## Case 1: Cluster provides NVHPC

On some clusters, the NVIDIA HPC SDK already contains cuTENSOR inside the NVHPC
tree. A typical setup looks like:

```bash
module load nvhpc-hpcx-cuda13/25.11
```

Then configure:

```bash
cmake .. \
  -DQLTEN_USE_GPU=ON \
  -DHP_NUMERIC_USE_MKL=ON
```

If CMake still cannot find cuTENSOR automatically, point it at the NVHPC
cuTENSOR target directory or set the exact include/library paths:

```bash
cmake .. \
  -DQLTEN_USE_GPU=ON \
  -DCUTENSOR_ROOT=/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/math_libs/13.0/targets/x86_64-linux
```

or

```bash
cmake .. \
  -DQLTEN_USE_GPU=ON \
  -DCUTENSOR_INCLUDE_DIR=/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/math_libs/13.0/targets/x86_64-linux/include \
  -DCUTENSOR_LIBRARY=/opt/nvidia/hpc_sdk/Linux_x86_64/25.11/math_libs/13.0/targets/x86_64-linux/lib/libcutensor.so
```

## Case 2: User installs cuTENSOR under `$HOME`

If the cluster has CUDA but does not package cuTENSOR, install it into your
home directory and point CMake at that user-local prefix.

Recommended CMake hint:

```bash
-DCUTENSOR_ROOT=$HOME/.local/usr
```

TensorToolkit's finder accepts layouts such as:

- `<prefix>/include` + `<prefix>/lib64`
- `<prefix>/include` + `<prefix>/lib`
- `<prefix>/include` + `<prefix>/lib/x86_64-linux-gnu`
- extracted package layouts under `<prefix>/usr/include` and `<prefix>/usr/lib*`

### Ubuntu / Debian-style local install

Example flow:

```bash
wget https://developer.download.nvidia.com/compute/cutensor/2.1.0/local_installers/cutensor-local-repo-ubuntu2204-2.1.0_1.0-1_amd64.deb
dpkg-deb -x cutensor-local-repo-ubuntu2204-2.1.0_1.0-1_amd64.deb $HOME/cutensor_install
cd $HOME/cutensor_install/var/cutensor-local-repo-ubuntu2204-2.1.0

mkdir -p $HOME/cutensor_install/libcutensor2
dpkg-deb -x libcutensor2_2.1.0.9-1_amd64.deb $HOME/cutensor_install/libcutensor2

mkdir -p $HOME/cutensor_install/libcutensor-dev
dpkg-deb -x libcutensor-dev_2.1.0.9-1_amd64.deb $HOME/cutensor_install/libcutensor-dev
```

Then configure with:

```bash
cmake .. \
  -DQLTEN_USE_GPU=ON \
  -DCUTENSOR_ROOT=$HOME/cutensor_install/libcutensor-dev/usr
```

If your package places libraries under `lib/x86_64-linux-gnu`, TensorToolkit's
finder should now pick that up directly. A `lib64` symlink is no longer
required for this project.

### CentOS / RHEL-style local install

Example flow:

```bash
wget https://developer.download.nvidia.com/compute/cutensor/2.0.2.1/local_installers/cutensor-local-repo-rhel8-2.0.2-1.0-1.x86_64.rpm
mkdir -p $HOME/cutensor_install
rpm2cpio cutensor-local-repo-rhel8-2.0.2-1.0-1.x86_64.rpm | cpio -idmv -D $HOME/cutensor_install

cd $HOME/cutensor_install/var/cutensor-local-repo-rhel8-2.0.2
rpm2cpio libcutensor2-*.rpm | cpio -idmv -D $HOME/.local
rpm2cpio libcutensor-devel-*.rpm | cpio -idmv -D $HOME/.local
```

Then configure with:

```bash
cmake .. \
  -DQLTEN_USE_GPU=ON \
  -DCUTENSOR_ROOT=$HOME/.local/usr
```

## Runtime environment

If cuTENSOR is installed in a non-standard user path, your runtime environment
may need to expose its libraries:

```bash
export LD_LIBRARY_PATH=$HOME/.local/usr/lib64:$LD_LIBRARY_PATH
```

or for Debian-style layouts:

```bash
export LD_LIBRARY_PATH=$HOME/cutensor_install/libcutensor-dev/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

You may also need to ensure CUDA itself is discoverable, for example with
`CUDAToolkit_ROOT=/path/to/cuda`.

## When auto-discovery still fails

Set the exact paths:

```bash
cmake .. \
  -DQLTEN_USE_GPU=ON \
  -DCUTENSOR_INCLUDE_DIR=/path/to/cutensor/include \
  -DCUTENSOR_LIBRARY=/path/to/libcutensor.so
```

This is the most portable fallback across clusters.
