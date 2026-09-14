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
