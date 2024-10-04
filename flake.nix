{
  description = "Set of parallel robot models for general use in benchmarks and examples";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:gepetto/nixpkgs/master";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = inputs.nixpkgs.lib.systems.flakeExposed;
      perSystem =
        { pkgs, self', ... }:
        {
          apps.default = {
            type = "app";
            program = pkgs.python3.withPackages (_: [ self'.packages.default ]);
          };
          devShells.default = pkgs.mkShell { inputsFrom = [ self'.packages.default ]; };
          packages = {
            default = self'.packages.example-parallel-robots;
            example-parallel-robots = pkgs.python3Packages.toPythonModule (pkgs.callPackage ./package.nix { });
          };
        };
    };
}
