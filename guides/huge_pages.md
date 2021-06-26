Huge Pages
==========

For GPU joins, huge pages serve two main purposes:

 - Reduce the number of TLB misses when accessing hash tables or partitioning
   data.

 - As a workaround to reserve memory so that the OS doesn't provide it to other
   applications.

In addition, the TLB latency microbenchmark relies on huge pages reserved at
early boot to avoid physical memory fragmentation.

## Memory fragmentation issues

The TLB latency microbenchmark is especially sensitive to memory fragmentation.
"Volta" GPUs load up to 16 page table entries at a time *if the pages are
physically adjacent*. Thus, avoiding fragmentation is very important for
measurement reproducibility!

## General huge pages guide

Here are some hints to working with huge pages on Linux:

 - Supported huge pages sizes: `ls /sys/kernel/mm/hugepages/`

 - Total 2 MB huge pages reserved in the system: `cat
   /sys/kernel/mm/hugepages/hugepages-2048kB/`

 - Total 2 MB huge pages reserved on NUMA node 0: `cat
   /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages`

 - Overcommitted huge pages, that Linux can allocate on-the-fly: `cat
   /sys/kernel/mm/hugepages/hugepages-2048kB/nr_overcommit_hugepages`

 - Reserve 10 x 2 MB huge pages on NUMA node 0: `sudo bash -c 'echo 10 >
   /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages'`

 - Show some system stats on memory usage: `numastat -m`

## Page sizes

The available page sizes depend on the CPU architecture, and on the operating
system configuration. On Ubuntu 18.04, the page sizes are:

 - x84_64: 4 KB small pages; 2 MB and 1 GB huge pages

 - ppc64le: 64 KB small pages; 2 MB and 1 GB huge pages (in radix MMU mode)

 - ppc64le: 64 KB small pages; 16 MB and 16 GB huge pages (in hash MMU mode)

Note that hash MMU mode is not supported by Nvidia GPUs (source: private
conversation with an Nvidia developer).

## Reserving 2 MB huge pages on early boot

Linux can be configured to reserve huge pages early in the boot process. This
section is adapted from the [Red Hat Enterprise Linux Performance Tuning
Guide](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/performance_tuning_guide/sect-red_hat_enterprise_linux-performance_tuning_guide-memory-configuring-huge-pages).

The following steps setup the reservation. These steps have been tested on
Ubuntu 18.04, but should work on other distributions as well (with minor
adaptations).

The configured amount of huge pages are enough to run a hash join with linear
probing (50% load factor) on two 2048 million 16-byte tuple relations. The
machine has 128 GiB per NUMA node.

**Warning**: Setting the number of pages higher than the physically available
memory can hang the boot process indefinitely. Leave at least 1 or 2 GB free
unless you have access to an IPMI console :-)

 1. Add the page reservation script `/lib/systemd/hugetlb-reserve-2M-pages.sh`:

    ```sh
    #!/bin/sh

    nodes_path=/sys/devices/system/node/
    if [ ! -d $nodes_path ]; then
            echo "ERROR: $nodes_path does not exist"
            exit 1
    fi

    reserve_pages()
    {
            echo $1 > $nodes_path/$2/hugepages/hugepages-2048kB/nr_hugepages
    }

    reserve_pages 64024 node0
    ```

 2. Add the systemd service `/lib/systemd/system/hugetlb-2M-pages.service`:

    ```sh
    [Unit]
    Description=HugeTLB 2M Pages Reservation
    DefaultDependencies=no
    Before=dev-hugepages.mount
    ConditionPathExists=/sys/devices/system/node
    # ConditionKernelCommandLine=hugepagesz=2M

    [Service]
    Type=oneshot
    RemainAfterExit=yes
    ExecStart=/lib/systemd/hugetlb-reserve-2M-pages.sh
    TimeoutSec=120
    Conflicts=hugetlb-1G-pages

    [Install]
    WantedBy=sysinit.target
    ```

 3. Set executable permissions on the reservation script:

    ```sh
    sudo chmod +x /lib/systemd/hugetlb-reserve-2M-pages.sh
    ```

 4. Enable the systemd service:

    ```sh
    sudo systemctl enable hugetlb-2M-pages
    ```

## Reserving 1 GB huge pages on early boot

The scripts are analogous to the 2 MB huge pages.

 - Reservation script `/lib/systemd/hugetlb-reserve-1G-pages.sh`:

   ```sh
   #!/bin/sh

   nodes_path=/sys/devices/system/node/
   if [ ! -d $nodes_path ]; then
   echo "ERROR: $nodes_path does not exist"
   exit 1
   fi

   reserve_pages()
   {
       echo $1 > $nodes_path/$2/hugepages/hugepages-1048576kB/nr_hugepages
   }

   reserve_pages 126 node0
   ```

 - Systemd service `/lib/systemd/system/hugetlb-1G-pages.service`:

   ```sh
   [Unit]
   Description=HugeTLB 1G Pages Reservation
   DefaultDependencies=no
   Before=dev-hugepages.mount
   ConditionPathExists=/sys/devices/system/node
   # ConditionKernelCommandLine=hugepagesz=2M

   [Service]
   Type=oneshot
   RemainAfterExit=yes
   ExecStart=/lib/systemd/hugetlb-reserve-1G-pages.sh
   TimeoutSec=120
   Conflicts=hugetlb-2M-pages

   [Install]
   WantedBy=sysinit.target
   ```

## References

[Linux kernel HugeTLB pages](https://www.kernel.org/doc/html/latest/admin-guide/mm/hugetlbpage.html)

[Linux kernel transparent hugepage support](https://www.kernel.org/doc/html/latest/admin-guide/mm/transhuge.html)
