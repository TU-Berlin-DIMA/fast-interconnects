with import <nixpkgs> {};

let src = fetchFromGitHub {
      owner = "mozilla";
      repo = "nixpkgs-mozilla";
      # commit from: 2019-05-15
      rev = "f233fdc4ff6ba2ffeb1e3e3cd6d63bb1297d6996";
      sha256 = "1rzz03h0b38l5sg61rmfvzpbmbd5fn2jsi1ccvq22rb76s1nbh8i";
   };
in
with import "${src.out}/rust-overlay.nix" pkgs pkgs;

gcc8Stdenv.mkDerivation {
  name = "rust-env";
  nativeBuildInputs = [
    # Rust overlay packages
    latest.rustChannels.stable.rust

    # NixPkgs packages
    clang numactl cudatoolkit_10_2 linuxPackages.nvidia_x11 valgrind
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
