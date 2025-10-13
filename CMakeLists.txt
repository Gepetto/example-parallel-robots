cmake_minimum_required(VERSION 3.22)

# Project properties
set(PROJECT_ORG gepetto)
set(PROJECT_NAME example-parallel-robots)
set(PROJECT_DESCRIPTION
    "Set of parallel robot models for general use in benchmarks and examples")
set(PROJECT_URL "https://github.com/${PROJECT_ORG}/${PROJECT_NAME}")

# Project configuration
set(CUSTOM_HEADER_DIR ${PROJECT_NAME})
set(PROJECT_COMPATIBILITY_VERSION AnyNewerVersion)

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(EXISTS "${JRL_CMAKE_MODULES}/base.cmake")
  message(STATUS "JRL cmakemodules found in 'cmake/' git submodule")
else()
  find_package(jrl-cmakemodules QUIET CONFIG)
  if(jrl-cmakemodules_FOUND)
    get_property(
      JRL_CMAKE_MODULES
      TARGET jrl-cmakemodules::jrl-cmakemodules
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "JRL cmakemodules found on system at ${JRL_CMAKE_MODULES}")
  else()
    message(STATUS "JRL cmakemodules not found. Let's fetch it.")
    include(FetchContent)
    FetchContent_Declare(
      "jrl-cmakemodules"
      GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git")
    FetchContent_MakeAvailable("jrl-cmakemodules")
    FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
  endif()
endif()

# JRL-cmakemodule setup
include("${JRL_CMAKE_MODULES}/base.cmake")

# Print initial message
message(STATUS "${PROJECT_DESCRIPTION}, version ${PROJECT_VERSION}")
message(STATUS "Copyright (C) 2023-2024 LAAS-CNRS")
message(STATUS "All rights reserved.")
message(STATUS "Released under the BSD 3-Clause License.")

# Project definition
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

if(BUILD_PYTHON_INTERFACE)
  # add_project_dependency(numpy REQUIRED)
  # add_project_dependency(toolbox-parallel-robots REQUIRED)
  include("${JRL_CMAKE_MODULES}/python.cmake")
  add_project_dependency(pinocchio REQUIRED)
  string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})

  set(${PROJECT_NAME}_PYTHON_FILES
      __init__.py __main__.py loader_tools.py robot_options.py talos_closed.py
      talos_closed_6d.py)

  foreach(python ${${PROJECT_NAME}_PYTHON_FILES})
    python_install_on_site(${PY_NAME} ${python})
  endforeach(python ${${PROJECT_NAME}_PYTHON_FILES})

  configure_file(${PY_NAME}/path.py.in ${PY_NAME}/path.py)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PY_NAME}/path.py
          DESTINATION "${PYTHON_SITELIB}/${PY_NAME}")

  if(BUILD_TESTING)
    find_package(toolbox-parallel-robots QUIET) # tpr installed from source /
                                                # system
    if(NOT TARGET toolbox-parallel-robots::toolbox-parallel-robots)
      # tpr installed from PyPI
      execute_process(
        COMMAND ${PYTHON_EXECUTABLE} "-c" "import toolbox_parallel_robots"
        RESULT_VARIABLE _tpr
        OUTPUT_QUIET ERROR_QUIET)
      if(NOT _tpr EQUAL 0)
        message(
          FATAL_ERROR
            "toolbox-parallel-robots not found. Please install it and ensure it is available."
        )
      endif()
    endif()
    add_subdirectory(unittest)
  endif(BUILD_TESTING)
endif(BUILD_PYTHON_INTERFACE)

if(NOT BUILD_STANDALONE_PYTHON_INTERFACE)
  add_library(${PROJECT_NAME} INTERFACE)
  target_include_directories(${PROJECT_NAME}
                             INTERFACE $<INSTALL_INTERFACE:include>)
  target_compile_definitions(
    ${PROJECT_NAME}
    INTERFACE
      EXAMPLE_PARALLEL_ROBOTS_MODEL_DIR="$<INSTALL_PREFIX>/share/${PROJECT_NAME}/robots"
  )
  install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${TARGETS_EXPORT_NAME}
    DESTINATION lib)

  install(DIRECTORY robots DESTINATION share/${PROJECT_NAME})
endif()
