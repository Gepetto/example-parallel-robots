{
  description = "Set of parallel robot models for general use in benchmarks and examples";

  inputs = {
    gepetto.url = "github:gepetto/nix";
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
          { gepetto-pkgs.overlays = [ self.overlays.default ]; }
        ];
        flake.overlays.default = _final: prev: {
          pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
            (_python-final: python-prev: {
              example-parallel-robots = python-prev.example-parallel-robots.overrideAttrs {
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
            })
          ];
        };
        perSystem =
          { pkgs, self', ... }:
          {
            apps.default = {
              type = "app";
              program = pkgs.python3.withPackages (_: [ self'.packages.default ]);
            };
            packages = {
              default = self'.packages.example-parallel-robots;
              example-parallel-robots = pkgs.python3Packages.example-parallel-robots;
            };
          };
      }
    );
}
