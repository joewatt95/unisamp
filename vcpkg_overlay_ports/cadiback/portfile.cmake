vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/cadiback
  REF cdebc75b308149453f1e54d0dafe084caa2fee34
  SHA512 de84dd524cf1217c10e40f7364c8e08ab73ef11d403164d4c7f7ff70800f7f2e163284342d2421e6acfc7b3216914aee43cd14eee9b0a050b84f719c6c788385
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