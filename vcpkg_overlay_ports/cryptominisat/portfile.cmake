# include($ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO msoos/cryptominisat
  REF e60d03528e7bb2ad577f3ca5a8e6e478fe7407f1
  SHA512 11f0e50d1ab8981be74051b671ae1a62c8b8869fb59cb7fe5bf931b69174beca011d40a7de5956cff4a924b603d69dd0455b140a909ffce2ccdff85302391fb6
  HEAD_REF master
  # PATCHES
  #   changes.patch
)

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS
    -DCMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
)

vcpkg_cmake_install()