{
  description = "Set of parallel robot models for general use in benchmarks and examples";

  inputs.gepetto.url = "github:gepetto/nix";

  outputs =
    inputs:
    inputs.gepetto.lib.mkFlakoboros inputs (
      { lib, ... }:
      {
        pyOverrideAttrs.example-parallel-robots = {
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
    );
}
