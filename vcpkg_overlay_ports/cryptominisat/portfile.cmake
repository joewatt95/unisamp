vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO msoos/cryptominisat
  REF 0ab1e4406edc952a975347e7f3365f28d01f7f67
  SHA512 2a9c0628c09fd580c3a16c21e4c78635e4a8a142701ed216dfd9da6f60bc86190c561aee30632bb86e0248ea2bf445e9b1405e7ea2a189916db54dca13841d6b
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
  OPTIONS
    -DCMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
)

vcpkg_cmake_install()