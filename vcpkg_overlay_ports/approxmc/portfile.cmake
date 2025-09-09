vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/approxmc
  REF 27740ace7d40c2047530e4aff5d14ce4218ed2c2
  SHA512 13e9cc82725d8f5e41eda1cfd7213294def014a41f84181c1c6842a407f3ec5af0396c0e1502e875a96e86adf0478f75db4c135bf848327947e1fc25cd373067
)

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS
    -DCMAKE_TOOLCHAIN_FILE="${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
)

vcpkg_cmake_install()