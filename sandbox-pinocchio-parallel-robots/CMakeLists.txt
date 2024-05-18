cmake_minimum_required(VERSION 3.10)

# Project properties
set(PROJECT_ORG gepetto)
set(PROJECT_NAME sandbox-pinocchio-parallel-robots)
set(PROJECT_DESCRIPTION
    "Set of tools for general purpose operations on pinocchio models. This was originally designed to work with parallel robots."
)
set(PROJECT_URL https://gitlab.laas.fr/ldematteis/example-parallel-robots)

# Project options * Since the project is Python only for now, we remove the
# option of building the python interface option(BUILD_PYTHON_INTERFACE "Build
# the Python unit tests and helpers" ON)
set(BUILD_PYTHON_INTERFACE ON)
# option(INSTALL_PYTHON_INTERFACE_ONLY "Install *ONLY* the Python interface"
# OFF) option(BUILD_TESTING "Build unit tests" OFF)
set(BUILD_TESTING ON)

# Project configuration
set(CUSTOM_HEADER_DIR ${PROJECT_NAME})
set(PROJECT_COMPATIBILITY_VERSION AnyNewerVersion)

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/cmake/base.cmake")
  if(${CMAKE_VERSION} VERSION_LESS "3.14.0")
    message(
      FATAL_ERROR
        "\nPlease run the following command first:\ngit submodule update --init\n"
    )
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
# message(STATUS "Released under the BSD 3-Clause License.")

# Project definition
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

if(BUILD_PYTHON_INTERFACE)
  include("${JRL_CMAKE_MODULES}/python.cmake")
  add_project_dependency(pinocchio REQUIRED)
  string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})

  set(${PROJECT_NAME}_PYTHON_FILES __init__.py reorganize_models.py)
  foreach(python ${${PROJECT_NAME}_PYTHON_FILES})
    python_install_on_site(${PY_NAME} ${python})
  endforeach(python ${${PROJECT_NAME}_PYTHON_FILES})

endif(BUILD_PYTHON_INTERFACE)

if(BUILD_TESTING)
  add_subdirectory(unittest)
endif(BUILD_TESTING)

install(FILES package.xml DESTINATION share/${PROJECT_NAME})
