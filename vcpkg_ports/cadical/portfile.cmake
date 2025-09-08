vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO meelgroup/cadical
    REF 19b73b36ab9a0be427985abfb599be2da454225c
    SHA512 0
    HEAD_REF add_dynamic_lib
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