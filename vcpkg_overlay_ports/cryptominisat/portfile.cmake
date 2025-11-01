vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO msoos/cryptominisat
  REF cd1b4e1218a812b1a7c8f4780fad1f9ec5bf8b4b
  SHA512 ba3a9b98a62f1375e0c80cbf90c8c0992683b7476439a34e3cbb5c813ffc3839118f8574e649a44eb4c59cb40704cd43553f04bd68cfe5c93b196c9e91df0f3d
  PATCHES
    use_vcpkg_deps.patch
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
)

vcpkg_cmake_install()