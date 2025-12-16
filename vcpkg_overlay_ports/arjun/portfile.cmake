vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/arjun
  REF 2334ce5fd4c08c7dfc8f5a28ac1296dc119e85db
  SHA512 8a6aecbf8ef1ef45c5b650ba28513f9d7bddef9d6e7be9dcf676583ca547911623ddf73f0cf82bd15d2d0c4b0ba988f4852d9ad385967eb9af4d844a3d81e9a6
  PATCHES
    use_vcpkg_deps.patch
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
)

vcpkg_cmake_install()