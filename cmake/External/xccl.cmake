# Defines the torch::xccl interface target (oneCCL headers + libccl).
#
# third_party/torch-xpu-ops carries a copy of this logic for its own standalone
# builds. Both files share the __XCCL_INCLUDED guard, and this one runs first
# (from Dependencies.cmake, before the torch-xpu-ops subdirectory is added), so
# the in-tree definition wins and the version floor below is what is enforced.
if(NOT __XCCL_INCLUDED)
  set(__XCCL_INCLUDED TRUE)

  # Floor: the oldest oneCCL verified to build the xccl2 backend, not
  # necessarily the oldest that ships the onecclXxx C API XcclApi wraps. API
  # coverage varies between releases (onecclCommAbort, for one, is declared but
  # not implemented), so raise this only with evidence.
  set(TORCH_XCCL_MINIMUM_VERSION 2021.17)

  # XCCL_ROOT, XCCL_LIBRARY_DIR, XCCL_INCLUDE_DIR are handled by FindXCCL.cmake.
  find_package(XCCL ${TORCH_XCCL_MINIMUM_VERSION})
  if(NOT XCCL_FOUND)
    set(PYTORCH_FOUND_XCCL FALSE)
    # FindXCCL only sets a message for its own checks; a version below the
    # floor is reported by find_package_handle_standard_args instead.
    if(XCCL_NOT_FOUND_MESSAGE)
      message(WARNING "${XCCL_NOT_FOUND_MESSAGE}")
    else()
      message(WARNING "oneCCL ${TORCH_XCCL_MINIMUM_VERSION} or newer not found")
    endif()
  else()
    set(PYTORCH_FOUND_XCCL TRUE)
    if(NOT TARGET torch::xccl)
      add_library(torch::xccl INTERFACE IMPORTED)
      set_property(
        TARGET torch::xccl PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        ${XCCL_INCLUDE_DIR})
      # oneCCL declares every onecclXxx entry point __attribute__((weak)), so a
      # reference to one does not mark libccl.so.2 as used: --as-needed drops it
      # from DT_NEEDED, the calls bind to NULL, and the first collective
      # segfaults rather than the library failing to load. Drop the wrapper once
      # the weak attribute goes away upstream. CMAKE_DL_LIBS is kept from the
      # torch-xpu-ops copy, whose backend dlopen()s libccl.so.1 in
      # src/xccl/xccl.cpp and links through this target too.
      set_property(
        TARGET torch::xccl PROPERTY INTERFACE_LINK_LIBRARIES
        "-Wl,--no-as-needed,${XCCL_LIBRARY},--as-needed" ${CMAKE_DL_LIBS})
    endif()
  endif()
endif()
