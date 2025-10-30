vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/sbva
  REF a4d115e1217c40a95bb06bd642aca40d2cee465e
  SHA512 4319ff42161715fd530a5464af35b9b2dc87f36474d79ba8f5339f2d3d91d0b202eb01468ef6ab7c581b1e959502ca9e7fe47a283e9d6138c90ea1538d8bad03
  PATCHES
    use_vcpkg_deps.patch
    cassert.patch
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
)

vcpkg_cmake_install()
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")

# file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

# file(
#   COPY "${CURRENT_PACKAGES_DIR}/share"
#   DESTINATION "${CURRENT_PACKAGES_DIR}/debug"
# )

# vcpkg_cmake_config_fixup()