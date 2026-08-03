# Find the oneCCL (XCCL) runtime used by the c10d XCCL backends.
#
# This will define the following variables:
#   XCCL_FOUND        : True if the system has the XCCL library.
#   XCCL_INCLUDE_DIR  : Include directories needed to use XCCL.
#   XCCL_LIBRARY_DIR  : The path to the XCCL library directory.
#   XCCL_LIBRARY      : libccl.so.2.0, the `onecclXxx` C API.
#   XCCL_VERSION      : ONECCL_MAJOR.ONECCL_MINOR.ONECCL_PATCH from ccl.h.
#
# Both c10d XCCL backends, xccl2 and the torch-xpu-ops one, use only the C API,
# so this matches third_party/torch-xpu-ops/cmake/Modules/FindXCCL.cmake and
# does not look for libccl.so.1 (the C++ `ccl::` API).

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(XCCL_FOUND False)
  set(XCCL_NOT_FOUND_MESSAGE "OneCCL library is only supported on Linux!")
  return()
endif()

# The oneCCL environment (CCL_ROOT) must be sourced before building.
set(XCCL_ROOT $ENV{CCL_ROOT})

find_file(
  XCCL_INCLUDE_DIR
  NAMES include
  HINTS ${XCCL_ROOT}
  NO_DEFAULT_PATH
)

find_file(
  XCCL_INCLUDE_ONEAPI_DIR
  NAMES oneapi
  HINTS ${XCCL_ROOT}/include/
  NO_DEFAULT_PATH
)

find_file(
  XCCL_LIBRARY_DIR
  NAMES lib
  HINTS ${XCCL_ROOT}
  NO_DEFAULT_PATH
)

find_library(
  XCCL_LIBRARY
  NAMES libccl.so.2.0
  HINTS ${XCCL_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

if((NOT XCCL_INCLUDE_DIR) OR (NOT XCCL_INCLUDE_ONEAPI_DIR) OR (NOT XCCL_LIBRARY_DIR) OR (NOT XCCL_LIBRARY))
  set(XCCL_FOUND False)
  set(XCCL_NOT_FOUND_MESSAGE "OneCCL library not found!!")
  return()
endif()

set(XCCL_VERSION "")
if(EXISTS "${XCCL_INCLUDE_ONEAPI_DIR}/ccl.h")
  file(STRINGS "${XCCL_INCLUDE_ONEAPI_DIR}/ccl.h" XCCL_VERSION_DEFINES
       REGEX "^#define ONECCL_(MAJOR|MINOR|PATCH) ")
  foreach(_component MAJOR MINOR PATCH)
    string(REGEX MATCH "#define ONECCL_${_component} ([0-9]+)"
           _matched "${XCCL_VERSION_DEFINES}")
    if(_matched)
      set(XCCL_VERSION_${_component} ${CMAKE_MATCH_1})
    endif()
  endforeach()
  if(DEFINED XCCL_VERSION_MAJOR AND DEFINED XCCL_VERSION_MINOR AND DEFINED XCCL_VERSION_PATCH)
    set(XCCL_VERSION "${XCCL_VERSION_MAJOR}.${XCCL_VERSION_MINOR}.${XCCL_VERSION_PATCH}")
  endif()
endif()

list(APPEND XCCL_INCLUDE_DIR ${XCCL_INCLUDE_ONEAPI_DIR})

SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH}
  "${XCCL_INCLUDE_DIR}")
SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
  "${XCCL_LIBRARY_DIR}")

find_package_handle_standard_args(
  XCCL
  FOUND_VAR XCCL_FOUND
  REQUIRED_VARS XCCL_INCLUDE_DIR XCCL_LIBRARY_DIR XCCL_LIBRARY
  VERSION_VAR XCCL_VERSION
  REASON_FAILURE_MESSAGE "${XCCL_NOT_FOUND_MESSAGE}"
)
