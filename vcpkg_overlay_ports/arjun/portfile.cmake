vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/arjun
  REF 58ec9aff687c9adcd6a26f158a947c07794e43f6
  SHA512 bb8865744a83ab2d8f7e9fbf37d8c83bd1eaa781b3582346c5d3fc60bd0fcee34b04ad8fd7db9e9814b7e5347b54509ed8f33eb1ccf178ad4f69e9605559d6d8
  PATCHES
    use_vcpkg_deps.patch
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
)

vcpkg_cmake_install()