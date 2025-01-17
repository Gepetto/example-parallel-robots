{
  description = "Set of parallel robot models for general use in benchmarks and examples";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:gepetto/nixpkgs/master";
    toolbox-parallel-robots = {
      url = "github:Gepetto/toolbox-parallel-robots";
      inputs.flake-parts.follows = "flake-parts";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = inputs.nixpkgs.lib.systems.flakeExposed;
      perSystem =
        { pkgs, self', system, ... }:
        {
          apps.default = {
            type = "app";
            program = pkgs.python3.withPackages (_: [ self'.packages.default ]);
          };
          devShells.default = pkgs.mkShell { inputsFrom = [ self'.packages.default ]; };
          packages = {
            default = self'.packages.example-parallel-robots;
            example-parallel-robots = pkgs.python3Packages.callPackage ./default.nix {
              inherit (inputs.toolbox-parallel-robots.packages.${system}) toolbox-parallel-robots;
            };
          };
        };
    };
}
