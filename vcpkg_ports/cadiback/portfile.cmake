vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO meelgroup/caback
    REF f3b1b21e99c48d0b2e09ef6af3451e621d94cad6
    SHA512 0
    HEAD_REF synthesis
    # PATCHES
        # ... patches ...
)

vcpkg_make_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    # OPTIONS
    #     -DBUILD_EXAMPLES=OFF
    #     -DBUILD_TESTS=OFF
)

vcpkg_make_install()