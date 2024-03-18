cmake_minimum_required(VERSION 3.10)

# Project properties
set(PROJECT_ORG gepetto)
set(PROJECT_NAME closed-loop-tools)
set(PROJECT_DESCRIPTION
    "Toolbox for parallel robots applications")
set(PROJECT_URL https://gitlab.laas.fr/ldematteis/example-parallel-robots)

# Project options
option(BUILD_PYTHON_INTERFACE "Build the Python unit tests and helpers" ON)
option(INSTALL_PYTHON_INTERFACE_ONLY "Install *ONLY* the Python interface" OFF)
option(BUILD_TESTING "Build unit tests" OFF)

# Project configuration
if(NOT INSTALL_PYTHON_INTERFACE_ONLY)
  set(PROJECT_USE_CMAKE_EXPORT TRUE)
endif(NOT INSTALL_PYTHON_INTERFACE_ONLY)
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
  add_project_dependency(eigenpy 2.7.11 REQUIRED)
  include("${JRL_CMAKE_MODULES}/python.cmake")
  add_project_dependency(pinocchio REQUIRED)
  string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})
  add_subdirectory(python)
  if(BUILD_TESTING)
    add_subdirectory(unittest)
  endif(BUILD_TESTING)
endif(BUILD_PYTHON_INTERFACE)

if(NOT INSTALL_PYTHON_INTERFACE_ONLY)
  add_library(${PROJECT_NAME} INTERFACE)
  target_include_directories(${PROJECT_NAME}
                             INTERFACE $<INSTALL_INTERFACE:include>)
  install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${TARGETS_EXPORT_NAME}
    DESTINATION lib)

#   install(FILES include/${CUSTOM_HEADER_DIR}/path.hpp
#           DESTINATION include/${CUSTOM_HEADER_DIR})
  install(FILES package.xml DESTINATION share/${PROJECT_NAME})
endif(NOT INSTALL_PYTHON_INTERFACE_ONLY)