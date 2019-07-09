extern crate accel;

use self::accel::device::{ComputeMode, Device};

pub fn print_current_device_information() {
    let device = Device::current().expect("Couldn't get current CUDA device");

    let property_map = device
        .get_property()
        .expect("Couldn't get device properties from CUDA");

    let device_name = device.name().expect("Couldn't get device name from CUDA");

    let device_cores = device.cores().expect("Couldn't get cores from CUDA");

    let device_flops = device.flops().expect("Couldn't get flops from CUDA") / 10_f64.powf(9.0);

    let compute_mode_str = match device
        .compute_mode()
        .expect("Couldn't get compute mode from CUDA")
    {
        ComputeMode::cudaComputeModeDefault => "Default",
        ComputeMode::cudaComputeModeExclusive => "Exclusive",
        ComputeMode::cudaComputeModeProhibited => "Prohibited",
        ComputeMode::cudaComputeModeExclusiveProcess => "ExclusiveProcess",
    };

    println!(
        r#"name: {name}
cores: {cores}
GFLOPS: {flops}
totalGlobalMem: {totalGlobalMem}
sharedMemPerBlock: {sharedMemPerBlock}
regsPerBlock: {regsPerBlock}
warpSize: {warpSize}"#,
        name = device_name,
        cores = device_cores,
        flops = device_flops,
        totalGlobalMem = property_map.totalGlobalMem,
        sharedMemPerBlock = property_map.sharedMemPerBlock,
        regsPerBlock = property_map.regsPerBlock,
        warpSize = property_map.warpSize,
    );

    println!(
        r#"memPitch: {memPitch}
maxThreadsPerBlock: {maxThreadsPerBlock}
maxThreadsDim: NA
maxGridSize: NA
clockRate: {clockRate}
totalConstMem: {totalConstMem}
major: {major}
minor: {minor}
textureAlignment: {textureAlignment}
texturePitchAlignment: {texturePitchAlignment}"#,
        memPitch = property_map.memPitch,
        maxThreadsPerBlock = property_map.maxThreadsPerBlock,
        // maxThreadsDim = property_map.maxThreadsDim,
        // maxGridSize = property_map.maxGridSize,
        clockRate = property_map.clockRate,
        totalConstMem = property_map.totalConstMem,
        major = property_map.major,
        minor = property_map.minor,
        textureAlignment = property_map.textureAlignment,
        texturePitchAlignment = property_map.texturePitchAlignment,
    );

    println!(
        r#"deviceOverlap: {deviceOverlap}
multiProcessorCount: {multiProcessorCount}
kernelExecTimeoutEnabled: {kernelExecTimeoutEnabled}
integrated: {integrated}
canMapHostMemory: {canMapHostMemory}
computeMode: {computeMode}
maxTexture1D: {maxTexture1D}
maxTexture1DMipmap: {maxTexture1DMipmap}
maxTexture1DLinear: {maxTexture1DLinear}
maxTexture2D: NA
maxTexture2DMipmap: NA
maxTexture2DLinear: NA
maxTexture2DGather: NA
maxTexture3D: NA
maxTexture3DAlt: NA
maxTextureCubemap: {maxTextureCubemap}
maxTexture1DLayered: NA
maxTexture2DLayered: NA
maxTextureCubemapLayered: NA"#,
        deviceOverlap = property_map.deviceOverlap,
        multiProcessorCount = property_map.multiProcessorCount,
        kernelExecTimeoutEnabled = property_map.kernelExecTimeoutEnabled,
        integrated = property_map.integrated,
        canMapHostMemory = property_map.canMapHostMemory,
        computeMode = compute_mode_str,
        maxTexture1D = property_map.maxTexture1D,
        maxTexture1DMipmap = property_map.maxTexture1DMipmap,
        maxTexture1DLinear = property_map.maxTexture1DLinear,
        // maxTexture2D = property_map.maxTexture2D,
        // maxTexture2DMipmap = property_map.maxTexture2DMipmap,
        // maxTexture2DLinear = property_map.maxTexture2DLinear,
        // maxTexture2DGather = property_map.maxTexture2DGather,
        // maxTexture3D = property_map.maxTexture3D,
        // maxTexture3DAlt = property_map.maxTexture3DAlt,
        maxTextureCubemap = property_map.maxTextureCubemap,
        // maxTexture1DLayered = property_map.maxTexture1DLayered,
        // maxTexture2DLayered = property_map.maxTexture2DLayered,
        // maxTextureCubemapLayered = property_map.maxTextureCubemapLayered,
    );

    println!(
        r#"maxSurface1D: {maxSurface1D}
maxSurface2D: NA
maxSurface3D: NA
maxSurface1DLayered: NA
maxSurface2DLayered: NA
maxSurfaceCubemap: {maxSurfaceCubemap}
maxSurfaceCubemapLayered: NA
surfaceAlignment: {surfaceAlignment}
concurrentKernels: {concurrentKernels}
ECCEnabled: {ECCEnabled}
pciBusID: {pciBusID}
pciDeviceID: {pciDeviceID}
pciDomainID: {pciDomainID}
tccDriver: {tccDriver}
asyncEngineCount: {asyncEngineCount}
unifiedAddressing: {unifiedAddressing}
memoryClockRate: {memoryClockRate}
memoryBusWidth: {memoryBusWidth}"#,
        maxSurface1D = property_map.maxSurface1D,
        // maxSurface2D = property_map.maxSurface2D,
        // maxSurface3D = property_map.maxSurface3D,
        // maxSurface1DLayered = property_map.maxSurface1DLayered,
        // maxSurface2DLayered = property_map.maxSurface2DLayered,
        maxSurfaceCubemap = property_map.maxSurfaceCubemap,
        // maxSurfaceCubemapLayered = property_map.maxSurfaceCubemapLayered,
        surfaceAlignment = property_map.surfaceAlignment,
        concurrentKernels = property_map.concurrentKernels,
        ECCEnabled = property_map.ECCEnabled,
        pciBusID = property_map.pciBusID,
        pciDeviceID = property_map.pciDeviceID,
        pciDomainID = property_map.pciDomainID,
        tccDriver = property_map.tccDriver,
        asyncEngineCount = property_map.asyncEngineCount,
        unifiedAddressing = property_map.unifiedAddressing,
        memoryClockRate = property_map.memoryClockRate,
        memoryBusWidth = property_map.memoryBusWidth,
    );

    println!(
        r#"l2CacheSize: {l2CacheSize}
maxThreadsPerMultiProcessor: {maxThreadsPerMultiProcessor}
streamPrioritiesSupported: {streamPrioritiesSupported}
globalL1CacheSupported: {globalL1CacheSupported}
localL1CacheSupported: {localL1CacheSupported}
sharedMemPerMultiprocessor: {sharedMemPerMultiprocessor}
regsPerMultiprocessor: {regsPerMultiprocessor}
managedMemory: {managedMemory}
isMultiGpuBoard: {isMultiGpuBoard}
multiGpuBoardGroupID: {multiGpuBoardGroupID}
hostNativeAtomicSupported: {hostNativeAtomicSupported}
singleToDoublePrecisionPerfRatio: {singleToDoublePrecisionPerfRatio}
pageableMemoryAccess: {pageableMemoryAccess}
concurrentManagedAccess: {concurrentManagedAccess}
computePreemptionSupported: {computePreemptionSupported}
canUseHostPointerForRegisteredMem: {canUseHostPointerForRegisteredMem}
cooperativeLaunch: {cooperativeLaunch}
cooperativeMultiDeviceLaunch: {cooperativeMultiDeviceLaunch}
sharedMemPerBlockOptin: {sharedMemPerBlockOptin}
pageableMemoryAccessUsesHostPageTables: {pageableMemoryAccessUsesHostPageTables}
directManagedMemAccessFromHost: {directManagedMemAccessFromHost}"#,
        l2CacheSize = property_map.l2CacheSize,
        maxThreadsPerMultiProcessor = property_map.maxThreadsPerMultiProcessor,
        streamPrioritiesSupported = property_map.streamPrioritiesSupported,
        globalL1CacheSupported = property_map.globalL1CacheSupported,
        localL1CacheSupported = property_map.localL1CacheSupported,
        sharedMemPerMultiprocessor = property_map.sharedMemPerMultiprocessor,
        regsPerMultiprocessor = property_map.regsPerMultiprocessor,
        managedMemory = property_map.managedMemory,
        isMultiGpuBoard = property_map.isMultiGpuBoard,
        multiGpuBoardGroupID = property_map.multiGpuBoardGroupID,
        hostNativeAtomicSupported = property_map.hostNativeAtomicSupported,
        singleToDoublePrecisionPerfRatio = property_map.singleToDoublePrecisionPerfRatio,
        pageableMemoryAccess = property_map.pageableMemoryAccess,
        concurrentManagedAccess = property_map.concurrentManagedAccess,
        computePreemptionSupported = property_map.computePreemptionSupported,
        canUseHostPointerForRegisteredMem = property_map.canUseHostPointerForRegisteredMem,
        cooperativeLaunch = property_map.cooperativeLaunch,
        cooperativeMultiDeviceLaunch = property_map.cooperativeMultiDeviceLaunch,
        sharedMemPerBlockOptin = property_map.sharedMemPerBlockOptin,
        pageableMemoryAccessUsesHostPageTables =
            property_map.pageableMemoryAccessUsesHostPageTables,
        directManagedMemAccessFromHost = property_map.directManagedMemAccessFromHost,
    );
}
