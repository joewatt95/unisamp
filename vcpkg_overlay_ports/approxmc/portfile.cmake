vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/approxmc
  REF 56042dc9002dee312bb4be283d2bdf8bc2a67827
  SHA512 893a695c8675594fd0c9d25d2a78c8d44c9aa102a2bb6ebfa6aa22ff54ac996d665d87a03843fdc56244efab38b9d88cffb882a5a1579abf296fd3b10c7c4893
  PATCHES
    use_vcpkg_deps.patch
)

vcpkg_cmake_configure(
  SOURCE_PATH ${SOURCE_PATH}
)

vcpkg_cmake_install()