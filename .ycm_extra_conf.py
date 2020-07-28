# Source: https://ops.tips/gists/navigating-the-linux-kernel-source-with-youcompleteme/
#
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

flags = [ '-x', 'cuda', '--cuda-path=/opt/cuda', '--cuda-gpu-arch=sm_70', '-std=c++11', '-Wall', '-Wextra', '-D__syncthreads()=', ]

include_dirs = [
  './sql-ops/include',
]

def Settings( **kwargs ):
    """
    Given a source file, retrieves the flags necessary for compiling it.
    """
    for dir in include_dirs:
        flags.append('-I' + os.path.join(CURRENT_DIR, dir))

    return { 'flags': flags }
