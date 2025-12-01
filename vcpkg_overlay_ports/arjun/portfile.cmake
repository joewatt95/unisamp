vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/arjun
  REF d5fff418c94db41ace21e59ee85044424106d9c4
  SHA512 119805780637f0f870c50787b186a33a8b27c2aded63f236d4f1a02ba52bb2eaefe3f1ad30448c78ffb7d3cf94933d7b3d39880b4ce2aa6da98712d2cc63f604
  PATCHES
    use_vcpkg_deps.patch
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
)

vcpkg_cmake_install()