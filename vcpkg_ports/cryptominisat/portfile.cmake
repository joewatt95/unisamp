vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO msoos/cryptominisat
    REF 5.13.0 # Specify your desired GitHub tag here
    SHA512 0
    HEAD_REF master
    # PATCHES
        # ... patches ...
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
      -DCMAKE_CFLAGS "${CMAKE_CFLAGS} -std=gnu17"
)

vcpkg_cmake_install()