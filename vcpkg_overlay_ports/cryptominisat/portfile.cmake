vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO msoos/cryptominisat
  REF b09bd6bf05253adf5981e44f9dbd374b2811ff94
  SHA512 567c3a320fbd18e40f05a579e7b67f5b1c2d226bb709075994e8ae0d161e59e44d7f263eae7b016b3070ae4ce0d80c6c79c33b35d6e1128848db206c78fc1b08
  PATCHES
    use_vcpkg_deps.patch
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
)

vcpkg_cmake_install()