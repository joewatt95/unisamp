vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/cadiback
  REF f3b1b21e99c48d0b2e09ef6af3451e621d94cad6
  SHA512 4b73a4208b6a01d3ac88ac7bb6b0775bc166dcab8448486a90c839083b452b0ca07ce13f3f490b8d6305d9e13d9dc485e4126bb998e248dda5f9f9704a39a325
  HEAD_REF synthesis
  PATCHES
    changes.patch
)

set(ENV{CADICAL_INSTALL_DIR} ${CURRENT_INSTALLED_DIR})
set(ENV{CADICAL_VERSION} "2.0.0")
set(ENV{CADICAL_GITID} "19b73b36ab9a0be427985abfb599be2da454225c")

# file(
#   INSTALL "${CURRENT_PACKAGES_DIR}/lib/libcadical.a"
#   DESTINATION "${CURRENT_BUILDTREES_DIR}/cadical/build"
# )

# file(
#   INSTALL "${CURRENT_PACKAGES_DIR}/include/cadical.hpp"
#   DESTINATION "${CURRENT_BUILDTREES_DIR}/cadical/src"
# )

# file(
#   WRITE "${CURRENT_BUILDTREES_DIR}/cadical/VERSION"
#   "2.0.0"
# )

vcpkg_make_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  COPY_SOURCE
  DISABLE_DEFAULT_OPTIONS
)

vcpkg_execute_required_process(
  COMMAND make
  WORKING_DIRECTORY "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel"
  LOGNAME "build-${TARGET_TRIPLET}-rel"
)

# vcpkg_build_make(
#   BUILD_TARGET "all"
#   MAKEFILE "makefile.in"
# )

file(
  INSTALL "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/libcadiback.a"
  DESTINATION "${CURRENT_PACKAGES_DIR}/lib"
)
file(
  INSTALL "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/libcadiback.a"
  DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib"
)

file(
  INSTALL "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/libcadiback.so"
  DESTINATION "${CURRENT_PACKAGES_DIR}/lib"
)
file(
  INSTALL "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/libcadiback.so"
  DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib"
)

file(
  INSTALL "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/include/cadiback.h"
  DESTINATION "${CURRENT_PACKAGES_DIR}/include"
)
file(
  INSTALL "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/include/cadiback.h"
  DESTINATION "${CURRENT_PACKAGES_DIR}/debug/include"
)

# file(REMOVE_RECURSE "${CURRENT_BUILDTREES_DIR}/cadical")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")

# vcpkg_make_install(
#   MAKEFILE "makefile.in"
#   TARGETS ""
# )