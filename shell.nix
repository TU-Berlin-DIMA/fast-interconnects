# Pin Nixpkgs version to Rust 1.60
with import (fetchTarball {
  name = "nixpkgs";
  url = "https://github.com/NixOS/nixpkgs/archive/61970a41ec4605b23558fadfc75a1560c2bddef4.tar.gz";
  sha256 = "092manddxwvjdk71mkq5xbrkd4n3s1993yjkzw0riclqz0558hy5";
}) {};

# Comment in to use system default version instead of pinned version
# with import <nixpkgs> {};

gcc8Stdenv.mkDerivation {
  name = "rust-env";
  nativeBuildInputs = [
    # Rust packages
    cargo clippy rustc rustfmt

    # NixPkgs packages
    clang numactl cudaPackages_10_2.cudatoolkit linuxPackages.nvidia_x11 valgrind
  ];
  buildInputs = [
    # Example Run-time Additional Dependencies
  ];

  # Set Environment Variables
  RUST_BACKTRACE = "full";
  CUDA_PATH=pkgs.cudaPackages_10_2.cudatoolkit;

  # LD_FLAGS = "-L${pkgs.cudaPackages_10_2.cudatoolkit}/lib -L${pkgs.cudaPackages_10_2.cudatoolkit.lib}/lib";
  RUSTFLAGS = "-Lnative=${pkgs.stdenv.cc.cc.lib}/lib -Lnative=${pkgs.cudaPackages_10_2.cudatoolkit}/lib -Lnative=${pkgs.cudaPackages_10_2.cudatoolkit.lib}/lib -Lnative=${pkgs.linuxPackages.nvidia_x11}/lib";
  LD_LIBRARY_PATH = "${pkgs.cudaPackages_10_2.cudatoolkit}/lib:${pkgs.cudaPackages_10_2.cudatoolkit}/lib/stubs:${pkgs.cudaPackages_10_2.cudatoolkit.lib}/lib:$LD_LIBRARY_PATH -Lnative=${pkgs.linuxPackages.nvidia_x11}/lib";
}
