/// The device type and it's ID
///
/// Used to specify where a task should be run.
/// For example, CPU ID for numactl or GPU ID for CUDA
#[derive(Debug, Clone, Serialize, Eq, PartialEq)]
pub enum DeviceId {
    Cpu(u16),
    Gpu(u32),
}

/// The memory buffer type and it's location
///
/// Used to specify where memory should be allocated before performing
/// the allocation.
/// For example, the numactl ID.
#[derive(Debug, Clone, Serialize, Eq, PartialEq)]
pub enum MemoryLocation {
    System(u16),
    Unified,
}

/// The memory allocation method
///
/// Used to specify how the memory is allocated. Pageable memory can be
/// allocated with the system memory allocator. In contrast, pinned memory must
/// either be allocated using the cudaHostAlloc function, or dynamically pinned
/// after allocation with the cudaHostRegister function.
#[derive(Debug, Clone, Serialize, Eq, PartialEq)]
pub enum MemoryAllocationType {
    Pageable,
    Pinned,
    DynamicallyPinned,
}

/// CUDA grid size
#[derive(Debug, Copy, Clone, Default, Serialize, Eq, Ord, PartialEq, PartialOrd)]
pub struct Grid(pub u32);

/// CUDA block size
#[derive(Debug, Copy, Clone, Default, Serialize, Eq, Ord, PartialEq, PartialOrd)]
pub struct Block(pub u32);

/// CUDA warp size
#[derive(Debug, Copy, Clone, Default, Serialize, Eq, Ord, PartialEq, PartialOrd)]
pub struct Warp(pub i32);

/// CUDA streaming multiprocessor count
#[derive(Debug, Copy, Clone, Default, Serialize, Eq, Ord, PartialEq, PartialOrd)]
pub struct SM(pub i32);

/// Oversubscription ratio
///
/// The oversubscription ratio specifies the number of work groups per streaming
/// multiprocessor.
#[derive(Debug, Copy, Clone, Default, Serialize, Eq, Ord, PartialEq, PartialOrd)]
pub struct OversubRatio(pub u32);

/// Multiplying the oversubscription ratio with the multiprocessor count
/// yields the grid size.
impl std::ops::Mul<SM> for OversubRatio {
    type Output = Grid;

    fn mul(self, rhs: SM) -> Grid {
        Grid(self.0 * rhs.0 as u32)
    }
}

/// Warp multiplier
///
/// The warp multiplier specifies the number of warps that a work group consists
/// of.
#[derive(Debug, Copy, Clone, Default, Serialize, Eq, Ord, PartialEq, PartialOrd)]
pub struct WarpMul(pub u32);

/// The warp multiplier times the warp size yields the block size.
impl std::ops::Mul<Warp> for WarpMul {
    type Output = Block;

    fn mul(self, rhs: Warp) -> Block {
        Block(self.0 * rhs.0 as u32)
    }
}

/// Instruction level parallelism
///
/// The instruction level parallelism specifies the number of (usually identical)
/// instructions each work item runs at the same time. This is similar to manual
/// loop unrolling.
///
/// See also [V. Volkov "Better Performance at Lower Occupancy"](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf).
#[derive(Debug, Copy, Clone, Default, Serialize, Eq, Ord, PartialEq, PartialOrd)]
pub struct Ilp(pub u32);

/// Thread count
///
/// The number of CPU threads.
#[derive(Copy, Clone, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ThreadCount(pub usize);
