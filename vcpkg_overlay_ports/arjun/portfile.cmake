vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/arjun
  REF 6c31a24f9e9d050a01fec2f6a5642c89b06bb5ea
  SHA512 41c44046c6860f58f4148f06d5d12d27b216dbf7b8323153d31cf29adac86d731c94959350ddae9883049d429f14c5f8fb4b33f070a73cc7a90651f6ce478fb4
  PATCHES
    use_vcpkg_deps.patch
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
)

vcpkg_cmake_install()