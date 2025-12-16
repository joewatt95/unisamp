vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/approxmc
  REF 6213fb2bdee9d59bb14e8364e50a44944bd5eee8
  SHA512 58a83d572326d3b24175b7189626b5a0f0691cd41d7314af21bde4c48a8e808a0e3a3381268a26d1c6e113b89a727039387e10758c14af9a60d3051c4a0f7830
  PATCHES
    use_vcpkg_deps.patch
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
)

vcpkg_cmake_install()