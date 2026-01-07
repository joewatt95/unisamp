vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/approxmc-cert
  REF 4845882923341f3070cd1f1e9e40b3f194a44c11
  SHA512 10398f239cdddf9c6267da6dfd7eb79f32872999bf3623834572bdb4e20cb00aebb22a98df4a25a68bd257a3e49c3868a1e7d41cfb17e0f4c3abee05907b5f44 
  PATCHES
    use_vcpkg_deps.patch
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
)

vcpkg_cmake_install()