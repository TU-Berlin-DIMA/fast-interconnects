extern crate libc;

use std::fmt;

pub struct ProcessorCache {}

impl ProcessorCache {
    #[allow(non_snake_case)]
    pub fn L1D_size() -> usize {
        let size = unsafe { libc::sysconf(libc::_SC_LEVEL1_DCACHE_SIZE) };
        if size == -1 {
            // TODO: std::Option
        }
        size as usize
    }

    #[allow(non_snake_case)]
    pub fn L2_size() -> usize {
        let size = unsafe { libc::sysconf(libc::_SC_LEVEL2_CACHE_SIZE) };
        size as usize
    }

    #[allow(non_snake_case)]
    pub fn L3_size() -> usize {
        let size = unsafe { libc::sysconf(libc::_SC_LEVEL3_CACHE_SIZE) };
        size as usize
    }

    pub fn page_size() -> usize {
        let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        size as usize
    }
}

impl fmt::Display for ProcessorCache {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "L1 cache size: {}\nL2 cache size: {}\nL3 cache size: {}\npage size: {}",
            Self::L1D_size(),
            Self::L2_size(),
            Self::L3_size(),
            Self::page_size()
        )
    }
}

#[cfg(target_arch = "x86_64")]
pub fn cpu_codename() -> String {
    extern crate raw_cpuid;
    let cpuid = raw_cpuid::CpuId::new();
    cpuid
        .get_extended_function_info()
        .as_ref()
        .and_then(|i| i.processor_brand_string())
        .map_or_else(|| String::from("unknown x86-64"), |s| String::from(s))
}

#[cfg(target_arch = "powerpc64")]
pub fn cpu_codename() -> String {
    String::from("POWER9")
}
