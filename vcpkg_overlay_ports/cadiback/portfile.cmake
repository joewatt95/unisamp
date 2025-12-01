vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/cadiback
  REF 6050733173995f3cff6ecb04ac99d3691bd0b5e4
  SHA512 ab901ce2558b0248b9231868ed5ce1805aec55dd39d09c3c32b19f75a64429e1224be37f5835e1e182186c1f01f2debb74c9b9dfa37012b18d9160abf10f01e4
  HEAD_REF synthesis
  PATCHES
    use_vcpkg_deps.patch
)

set(ENV{CADICAL_INSTALL_DIR} ${CURRENT_INSTALLED_DIR})
set(ENV{CADICAL_VERSION} "2.0.0")
set(ENV{CADICAL_GITID} "19b73b36ab9a0be427985abfb599be2da454225c")

vcpkg_make_configure(
  SOURCE_PATH ${SOURCE_PATH}
  COPY_SOURCE
  DISABLE_DEFAULT_OPTIONS
)

if(VCPKG_BUILD_TYPE MATCHES debug)
  set(BUILD_SUFFIX "dbg")
  set(DEST_DIR "${CURRENT_PACKAGES_DIR}/debug")
elseif(VCPKG_BUILD_TYPE MATCHES release)
  set(BUILD_SUFFIX "rel")
  set(DEST_DIR {CURRENT_PACKAGES_DIR})
else()
  set(BUILD_SUFFIX "dbg" "rel")
  set(DEST_DIR "${CURRENT_PACKAGES_DIR}/debug" ${CURRENT_PACKAGES_DIR})
endif()

if(VCPKG_LIBRARY_LINKAGE MATCHES static)
  set(LIB_FILE "libcadiback.a")
elseif(VCPKG_LIBRARY_LINKAGE MATCHES dynamic)
  set(LIB_FILE "libcadiback.so")
else()
  message(WARNING "VCPKG_LIBRARY_LINKAGE is not 'static' or 'dynamic'. Current value: ${VCPKG_LIBRARY_LINKAGE}")
endif()

foreach(build_suffix dest_dir IN ZIP_LISTS BUILD_SUFFIX DEST_DIR)
  vcpkg_execute_required_process(
    COMMAND make "-j8" ${LIB_FILE}
    WORKING_DIRECTORY "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-${build_suffix}"
    LOGNAME "build-${TARGET_TRIPLET}-${build_suffix}"
  )

  file(
    INSTALL "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-${build_suffix}/${LIB_FILE}"
    DESTINATION "${dest_dir}/lib"
  )

  file(
    INSTALL "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-${build_suffix}/include/cadiback.h"
    DESTINATION "${dest_dir}/include"
  )
endforeach()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")