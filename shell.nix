with import <nixpkgs> {};

gcc8Stdenv.mkDerivation {
  name = "rust-env";
  nativeBuildInputs = [
    rustc rustfmt cargo cargo-flamegraph clang numactl cudatoolkit_10_2 linuxPackages.nvidia_x11

    # Example Build-time Additional Dependencies
    # pkgconfig
  ];
  buildInputs = [
    # Example Run-time Additional Dependencies
  ];

  # Set Environment Variables
  RUST_BACKTRACE = "full";
  CUDA_PATH=pkgs.cudatoolkit_10_2;

  # LD_FLAGS = "-L${pkgs.cudatoolkit_10_2}/lib -L${pkgs.cudatoolkit_10_2.lib}/lib";
  RUSTFLAGS = "-Lnative=${pkgs.stdenv.cc.cc.lib}/lib -Lnative=${pkgs.cudatoolkit_10_2}/lib -Lnative=${pkgs.cudatoolkit_10_2.lib}/lib -Lnative=${pkgs.linuxPackages.nvidia_x11}/lib";
  LD_LIBRARY_PATH = "${pkgs.cudatoolkit_10_2}/lib:${pkgs.cudatoolkit_10_2}/lib/stubs:${pkgs.cudatoolkit_10_2.lib}/lib:$LD_LIBRARY_PATH -Lnative=${pkgs.linuxPackages.nvidia_x11}/lib";
}
