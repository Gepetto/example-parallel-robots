{
  description = "Set of parallel robot models for general use in benchmarks and examples";

  inputs = {
    gepetto.url = "github:gepetto/nix";
    flake-parts.follows = "gepetto/flake-parts";
    systems.follows = "gepetto/systems";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } (
      { lib, ... }:
      {
        systems = import inputs.systems;
        imports = [
          inputs.gepetto.flakeModule
          {
            flakoboros.pyOverrideAttrs.example-parallel-robots = _: _: {
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
      }
    );
}
