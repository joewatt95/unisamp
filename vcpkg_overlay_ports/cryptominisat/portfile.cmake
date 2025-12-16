vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO msoos/cryptominisat
  REF 4c377ecab94ca9e9d3b2348204fb0ffe27fe6dec
  SHA512 8c6208376c78305969beda95ed59b3367a524de0ced49b49a5e44b2d9bbac42f978305738f7c79ee7f5a9b0fd4a0f5da482459218b94ef79b6c385e78f59a12a
  PATCHES
    use_vcpkg_deps.patch
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
)

vcpkg_cmake_install()