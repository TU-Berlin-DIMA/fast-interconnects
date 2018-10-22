numa-gpu
========

## Usage

Before building, ensure that Rust and CUDA are installed on your system.

You can install an up-to-date version of Rust using [rustup](https://rustup.rs):
```sh
curl https://sh.rustup.rs -sSf | sh
```

Use cargo to build numa-gpu:
```sh
cargo build
```

## What is numa-gpu?

numa-gpu is on-going research into using Nvidia's new NVLink 2.0 technology to
accelerate in-memory data analytics. In contrast to PCI-e and NVLink 1.0,
NVLink 2.0 provides hardware address translation services that, in effect, make
the GPU a NUMA node (Non-Uniform Memory Access). This new feature will enable
higher performance, but requires a re-design of the database.

In this project, we rethink database design to take full advantage of NVLink 2.0.
Especially out-of-core queries, i.e., queries that use data sets larger than
GPU memory, will see speed-ups compared to previous interconnect technologies.

## Platforms

The following platforms are currently tested:

* `x86_64-unknown-linux-gnu`
* `powerpc64le-unknown-linux-gnu`

Note that NVLink is only available on IBM POWER platforms.
