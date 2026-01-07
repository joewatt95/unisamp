# Copyright (c) 2017, Mate Soos
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
# include_directories(${Boost_INCLUDE_DIRS})
# include_directories(${CRYPTOMINISAT5_INCLUDE_DIRS})
# include_directories(${APPROXMC_INCLUDE_DIRS})
# include_directories(${GMP_INCLUDE_DIRS})
# include_directories(${ARJUN_INCLUDE_DIRS})

if (NOT WIN32)
    add_cxx_flag_if_supported("-Wno-bitfield-constant-conversion")
    #add_cxx_flag_if_supported("-Wduplicated-cond")
    #add_cxx_flag_if_supported("-Wduplicated-branches")
    add_cxx_flag_if_supported("-Wlogical-op")
    add_cxx_flag_if_supported("-Wrestrict")
    add_cxx_flag_if_supported("-Wnull-dereference")
    add_cxx_flag_if_supported("-Wdouble-promotion")
    add_cxx_flag_if_supported("-Wshadow")
    add_cxx_flag_if_supported("-Wformat=2")
    add_cxx_flag_if_supported("-Wextra-semi")
    add_cxx_flag_if_supported("-pedantic")
    #add_cxx_flag_if_supported("-Wdeprecated")
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/GitSHA1.cpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/GitSHA1.cpp" @ONLY)

set(unisamp_lib_files
  sampler.cpp
  unisamp.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/GitSHA1.cpp
)

add_library(unisamp ${unisamp_lib_files})

target_link_libraries(unisamp PUBLIC
  ${CRYPTOMINISAT5_LIBRARIES}
  ${ARJUN_LIBRARIES}
  ${APPROXMC_LIBRARIES}
  # ${MPFR_LIBRARIES}
  # ${GMP_LIBRARY}
)

set_target_properties(unisamp PROPERTIES
  PUBLIC_HEADER "${unisamp_public_headers}"
  VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
  SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

add_executable(unisamp_dynamic main.cpp)
set(unisamp_exec_link_libs "unisamp")
IF (ZLIB_FOUND)
  set(unisamp_exec_link_libs ${unisamp_exec_link_libs} ${ZLIB_LIBRARY})
ENDIF()
target_link_libraries(unisamp_dynamic ${unisamp_exec_link_libs})

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set_target_properties(unisamp_dynamic PROPERTIES
      # OUTPUT_NAME unisamp
      RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
      INSTALL_RPATH_USE_LINK_PATH TRUE
      LINK_FLAGS "-s WASM=1 -s ALLOW_MEMORY_GROWTH=1 -s EXPORTED_RUNTIME_METHODS='[\"callMain\", \"ccall\", \"cwrap\", \"FS\", \"print\"]' -s FORCE_FILESYSTEM=1"
    )
else()
    set_target_properties(unisamp_dynamic PROPERTIES
      # OUTPUT_NAME unisamp
      RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
      INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

unisamp_add_public_header(unisamp ${CMAKE_CURRENT_SOURCE_DIR}/unisamp.h)

# -----------------------------------------------------------------------------
# Copy public headers into build directory include directory.
# The unisampConfig.cmake we generate in the build directory depends on
# this.
# -----------------------------------------------------------------------------
set(HEADER_DEST "${PROJECT_BINARY_DIR}/include/unisamp")
add_custom_target(CopyPublicHeaders ALL)
get_target_property(unisamp_public_headers unisamp PUBLIC_HEADER)
foreach(public_header ${unisamp_public_headers})
    get_filename_component(HEADER_NAME ${public_header} NAME)
    add_custom_command(TARGET CopyPublicHeaders PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${HEADER_DEST}"
        COMMAND ${CMAKE_COMMAND} -E echo "Copying ${HEADER_NAME} to ${HEADER_DEST}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${public_header} "${HEADER_DEST}"
    )
endforeach()

install(TARGETS unisamp
    EXPORT ${UNISAMP_EXPORT_NAME}
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/unisamp"
)

install(TARGETS unisamp_dynamic
    EXPORT ${UNISAMP_EXPORT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
