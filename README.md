Project Triton
==============

## What is Project Triton?

Project Triton is a research project that aims to scale data management on GPUs
to a large data size by utilizing fast interconnects. Fast interconnects such
as NVLink 2.0 provide GPUs with high-bandwidth, cache-coherent access to main
memory. Thus, we want to unlock higher DBMS query performance with this new
class of hardware!

In this project, we rethink database design to take full advantage of fast
interconnects. GPUs can store only several gigabytes of data in their on-board
memory, while current interconnect technologies (e.g., PCI Express) are too
slow to transfer data ad hoc to the GPU. In contrast, CPUs are able to access
terabytes of data in main memory. Thus, GPU-based systems run into a data
transfer bottleneck.

Fast interconnects provide a path towards querying large data volumes
"out-of-core" in main memory. The Triton Project explores the ways in which
database management systems can take advantage of fast interconnects to achieve
a high data volume scalability.

## Guides

We provide a series of guides to setup Project Triton on your hardware, and on
how we tuned our code for IBM POWER9 CPUs and Nvidia Volta GPUs:

 * [Setup Guide](./guides/setup.md)

 * [Problems when getting started](./guides/problems.md)

 * [Huge pages tuning](./guides/huge_pages.md)

 * [NUMA in the context of fast interconnects](./guides/numa.md)

 * [POWER9 Microarchitecture Tuning](./guides/power9.md)

## Code Structure

NUMA-GPU provides the following applications and libraries:

 * `datagen` is a application and library to generate data with data
   distributions. It is used as a library by `data-store` and `tpch-bench`.
 * `data-store` is a library for generating relational data sets. It is used by
   `hashjoin` and `radix-join`.
 * `hashjoin` is an application to execute and benchmark hash joins on CPUs and
   GPUs.
 * `microbench` is a collection of microbenchmarks for CPUs, GPUs, and GPU
   interconnects.
 * `numa-gpu` is a library with abstractions and tools to program GPUs with and
   without NVLink.
 * `radix-join` is an application to execute and benchmark radix joins on CPUs
   and GPUs. The distinction from `hashjoin` enables a specialized API for
   radix joins.
 * `sql-ops` is a library that implements SQL operators. These are used by
   `hashjoin` and `radix-join`.
 * `tpch-bench` is an application to execute and benchmark TPC-H on CPUs and
   GPUs. Currently, Query 6 is implemented.

[Code documentation is available on GitHub
Pages](https://tu-berlin-dima.github.io/numa-gpu/sql_ops/). You can also build
it yourself by running:
```sh
cargo doc --document-private-items --no-deps --open
```

## Research

We've published our results from the Triton Project as academic papers:

 * [Lutz et al., *Pump Up the Volume: Processing Large Data on GPUs with Fast
   Interconnects*, SIGMOD 2020](https://doi.org/10.1145/3318464.3389705)

 * Lutz et al., *Triton Join: Efficiently Scaling to a Large Join State on GPUs
   with Fast Interconnects*, SIGMOD 2022

To cite our works, add these BibTeX snippets to your bibliography:

```
@InProceedings{lutz:sigmod:2020,
  author        = {Clemens Lutz and Sebastian Bre{\ss} and Steffen Zeuch and
                  Tilmann Rabl and Volker Markl},
  title         = {Pump up the volume: {Processing} large data on {GPUs} with
                  fast interconnects},
  booktitle     = {{SIGMOD}},
  pages         = {1633--1649},
  publisher     = {{ACM}},
  address       = {New York, NY, USA},
  year          = {2020},
  doi           = {10.1145/3318464.3389705}
}

@InProceedings{lutz:sigmod:2022,
  author        = {Clemens Lutz and Sebastian Bre{\ss} and Steffen Zeuch and
                  Tilmann Rabl and Volker Markl},
  title         = {Triton join: {Efficiently} scaling to a large join state
                  on {GPUs} with fast interconnects},
  booktitle     = {{SIGMOD}},
  numpages      = {16},
  publisher     = {{ACM}},
  address       = {New York, NY, USA},
  year          = {2022},
  doi           = {10.1145/3514221.3517911},
  note          = {To be published}
}
```
