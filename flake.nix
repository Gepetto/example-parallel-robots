{
  description = "Set of parallel robot models for general use in benchmarks and examples";

  inputs = {
    gepetto.url = "github:gepetto/nix";
    gazebros2nix.follows = "gepetto/gazebros2nix";
    flake-parts.follows = "gepetto/flake-parts";
    nixpkgs.follows = "gepetto/nixpkgs";
    nix-ros-overlay.follows = "gepetto/nix-ros-overlay";
    systems.follows = "gepetto/systems";
    treefmt-nix.follows = "gepetto/treefmt-nix";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } (
      { lib, self, ... }:
      {
        systems = import inputs.systems;
        imports = [
          inputs.gepetto.flakeModule
          {
            gazebros2nix.pyOverrides.example-parallel-robots = _final: _python-final: {
              src = lib.fileset.toSource {
                root = ./.;
                fileset = lib.fileset.unions [
                  ./CMakeLists.txt
                  ./example_parallel_robots
                  ./package.xml
                  ./robots
                  ./unittest
                ];
              };
            };
          }
        ];
        perSystem =
          { pkgs, ... }:
          {
            apps.default = {
              type = "app";
              program = pkgs.python3.withPackages (p: [ p.example-parallel-robots ]);
            };
          };
      }
    );
}
