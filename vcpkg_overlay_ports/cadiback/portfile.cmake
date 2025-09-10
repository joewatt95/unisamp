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

vcpkg_make_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  COPY_SOURCE
  DISABLE_DEFAULT_OPTIONS
)

if(VCPKG_BUILD_TYPE STREQUAL "debug")
  set(BUILD_SUFFIX "dbg")
  set(DEST_DIR "${CURRENT_PACKAGES_DIR}/debug")
elseif(VCPKG_BUILD_TYPE STREQUAL "release")
  set(BUILD_SUFFIX "rel")
  set(DEST_DIR "${CURRENT_PACKAGES_DIR}")
else()
  set(BUILD_SUFFIX "dbg" "rel")
  set(DEST_DIR "${CURRENT_PACKAGES_DIR}/debug" ${CURRENT_PACKAGES_DIR})
endif()

if(VCPKG_LIBRARY_LINKAGE STREQUAL "static")
  set(LIB_FILE "libcadiback.a")
elseif(VCPKG_LIBRARY_LINKAGE STREQUAL "dynamic")
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