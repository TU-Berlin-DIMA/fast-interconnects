Problems When Getting Started
=============================

Getting Project Triton to run for the first time can be a bit difficult. This
is research code, and requires a specific system configuration to achieve the
best performance.

Here's a guide to solving some of the problems we've experienced.

## Project Triton fails with: SIGILL (Illegal instruction)

Most likely, you've attempted to run a GPU program with the wrong memory type
(on a non-NVLink machine). Try using `--mem-type device`.

`hashjoin` also
supports `--execution-method gpustream` to stream blocks of `system` and `numa`
memory to the GPU. Alternatively, `--execution-method unified` can be used
togther with `--rel-mem-type unified`.

## Project Triton fails with: SIGBUS (Misaligned address error)

This is usually an out-of-memory condition. Try the following:

 - If you're trying to use huge pages, double-check if you've configured huge
   pages correctly. At least one of these command should show a non-zero value:

   ```sh
   cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
   cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_overcommit_hugepages
   ```

 - If you're trying to benchmark very large data (i.e., near the memory limit
   of a NUMA node), then the OS might not have enough memory for you on the
   requested NUMA node.

   In this case, it helps to [reserve huge pages on early
   boot](guides/huge_pages.md), and then the program with `--page-type
   Huge2MB`.

 - If none of the above apply, you may have found a bug! In this case, let us
   know by opening an issue or pull request.

## Project Triton fails with: CudaError(NoBinaryForGpu)

Your device isn't set as a build target. First, find the correct gencode for your
device in [Arnon Shimoni's helpful blog post][gencodes].
Then, add the gencode to the gencode lists in `sql-ops/build.rs` and
`microbench/build.rs`.

## Project Triton fails with: mlock() failed with ENOMEM

You don't have permission to lock memory with `mlock`. Setup permissions for your
user with:
```sh
sudo cat << EOF >> /etc/security/limits.conf
username		soft	memlock		unlimited
username		hard	memlock		unlimited
EOF
```

Now, log out and log in again.

## Project Triton fails with: Error: Device memory not supported in this context!

You've attempted to run a CPU program with device memory. Try using
`--mem-type system` or `--mem-type numa`.

ATTENTION NVLink users: The CPU does *not* have access to `--mem-type device`!
Instead, use `--mem-type numa --mem-location X`, where `X` is the GPU's NUMA
node identifier. You can find the correct ID with:
```sh
numactl --hardware
```

[gencodes]: http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

