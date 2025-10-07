vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/approxmc
  REF dd3a4be2fff1b433a261630b583ed46f407624c1
  SHA512 05f3ddfe092312ff81313be1b378f656c1aad4e2a4447908f37a02e73196e5e1b43162db817d906ea700c0e1064a4258cbed0a1ded265ff970862b52d9b46ff1
  PATCHES
    use_vcpkg_deps.patch
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
)

vcpkg_cmake_install()