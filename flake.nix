{
  description = "MPI C development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            openmpi
            gcc
            gnumake
            pkg-config
          ];

          shellHook = ''
            echo "MPI dev shell ready."
            echo "  mpicc  : $(command -v mpicc)"
            echo "  mpirun : $(command -v mpirun)"
            mpicc --version | head -n 1
            mpirun --version | head -n 1
          '';
        };
      });
}
