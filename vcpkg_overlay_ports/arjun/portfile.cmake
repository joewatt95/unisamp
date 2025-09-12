vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/arjun
  REF 9130e583a82cfec1cf12c198dc6fdf1a64e39481
  SHA512 daa246ca3766a87804013c54e965ef09cdfd2f775e94b7f75099369c83ba77989803a9b00a04b7985ee40eb171d7beee3a6ccae2740272298818518ac2fbf3fc
  PATCHES
    use_vcpkg_deps.patch
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
)

vcpkg_cmake_install()