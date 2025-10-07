vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/arjun
  REF d03f7c9c7857dc80af69bb874f9edec382f5fbc3
  SHA512 119c0bc49215b5340e232b2b632945d23beb6b5f11c5ad79fd60cc0ef88ba63b013aa939d9fb9ddb5ca4a3f5bab15836f2d1adb007c283334bd5bfe24da88cee
  PATCHES
    use_vcpkg_deps.patch
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
)

vcpkg_cmake_install()