set(${PROJECT_NAME}_PYTHON_TESTS robot)

# Tests require toolbox_parallel_robots
include(FetchContent)
FetchContent_Declare(
  "toolbox"
  GIT_REPOSITORY "https://github.com/gepetto/toolbox-parallel-robots"
  GIT_TAG "main")
FetchContent_Populate("toolbox")

foreach(TEST ${${PROJECT_NAME}_PYTHON_TESTS})
  add_python_unit_test("py-${TEST}" "unittest/test_${TEST}.py"
                       ".:${toolbox_SOURCE_DIR}")
endforeach(TEST ${${PROJECT_NAME}_PYTHON_TESTS})
