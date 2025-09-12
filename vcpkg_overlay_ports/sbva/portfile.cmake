vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/sbva
  REF 0faa08cf3cc26ed855831c9dc16a3489c9ae010f
  SHA512 c9f6fb54b5e6d201f5acf71472d4eeee4952a39f14c3289fa94e920077b98f8699c83090821789278bc749b56e2c5945397de615bbce1c8db707fac5b87ed1cb
  PATCHES
    use_vcpkg_deps.patch
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