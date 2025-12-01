vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/sbva
  REF 65993dbc0a802c0f1f7c83fac514339227f27b71
  SHA512 23c3174d59207312096d3b21aec2b9dbe8c61063af5ee9ffc18c338eaf1f8db8ab4023f5375b12fe2a69cd7c59f26e28fb04bdf358a733eebe72074907e27e72
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