{ pkgs, lib, config, inputs, ... }:

{
  env.GREET = "AfterBurner dev environment";

  packages = with pkgs; [
    git
    cmake
    clang
    wild
    pkg-config
    zlib
    openssl
    # For analysing run.jsonl. This is analysis tooling, not part of the crate:
    # the library and every shipped tool stay pure Rust, because a number in
    # research/findings.md should be reproducible by `cargo run` and not depend
    # on a second language being present. Reach for it when a question needs
    # reshaping a log and jq is not enough.
    python3
  ] ++ lib.optionals pkgs.stdenv.isLinux (with pkgs; [
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
  ]);

  languages.rust = {
    enable = true;
    toolchainFile = ./rust-toolchain.toml;
  };

  env.LD_LIBRARY_PATH = lib.makeLibraryPath ([
    pkgs.zlib
    pkgs.openssl
    pkgs.stdenv.cc.cc.lib
  ] ++ lib.optionals pkgs.stdenv.isLinux (with pkgs; [
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    libGL
    xorg.libX11
  ])) + ":/run/opengl-driver/lib";

  env.LIBRARY_PATH = "/run/opengl-driver/lib";

  env.CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";

  enterShell = ''
    echo "$GREET"
    rustc --version
    cargo --version
    echo "CUDA available: $(nvidia-smi &>/dev/null && echo yes || echo no)"
  '';
}
