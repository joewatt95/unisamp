vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO meelgroup/cadical
    REF 19b73b36ab9a0be427985abfb599be2da454225c
    SHA512 88b70b56f6785a15f79becc145794506d9784a16bc219cc2b4e821846e57e040bf0ed68a7f2ad82ca051f25b84dbd109536829dad89855789b0775095a734f24
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