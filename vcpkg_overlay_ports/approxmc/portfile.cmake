vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO meelgroup/approxmc
    REF 4.2.0
    SHA512 0
    HEAD_REF master
    # PATCHES
        # ... patches ...
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    # OPTIONS
    #     -DBUILD_EXAMPLES=OFF
    #     -DBUILD_TESTS=OFF
)

vcpkg_cmake_install()