vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO msoos/cryptominisat
  REF 34ec41b3bb0332bd313e51d131298d30c52aa71f
  SHA512 b5b02ebc44ba243fb29698e3fed7f08149cedf9153a2b4198ae51b5345cf02abc28e0b7c47d158c19284da87376ec6b860973acf5a77175cd3f5cb6d97b3f562
  PATCHES
    use_vcpkg_deps.patch
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
)

vcpkg_cmake_install()