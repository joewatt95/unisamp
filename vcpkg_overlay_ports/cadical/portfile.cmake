cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO meelgroup/cadical
  REF 07959ac34f139c247208c850da76087d4911b86a
  SHA512 38a8c5c18f9ec3ec6bb7a8a81aee6b693791b8178cc6f3994187ac2949499cbbce595786e91ecc827e44430c8ccf84a5de03efef83a5ec3a4a2d9112ce7ed7e4
)

set(VCPKG_C_FLAGS "${VCPKG_C_FLAGS} -fPIC")
set(VCPKG_CXX_FLAGS "${VCPKG_CXX_FLAGS} -fPIC")

vcpkg_make_configure(
  SOURCE_PATH ${SOURCE_PATH}
  COPY_SOURCE
  DISABLE_DEFAULT_OPTIONS
  OPTIONS
    --competition
)

if(VCPKG_BUILD_TYPE MATCHES debug)
  set(BUILD_SUFFIX "dbg")
  set(DEST_DIR "${CURRENT_PACKAGES_DIR}/debug")
elseif(VCPKG_BUILD_TYPE MATCHES release)
  set(BUILD_SUFFIX "rel")
  set(DEST_DIR ${CURRENT_PACKAGES_DIR})
else()
  set(BUILD_SUFFIX "dbg" "rel")
  set(DEST_DIR "${CURRENT_PACKAGES_DIR}/debug" ${CURRENT_PACKAGES_DIR})
endif()

if(VCPKG_LIBRARY_LINKAGE MATCHES static)
  set(LIB_FILE "libcadical.a")
elseif(VCPKG_LIBRARY_LINKAGE MATCHES dynamic)
  set(LIB_FILE "libcadical.so")
else()
  message(WARNING "VCPKG_LIBRARY_LINKAGE is not 'static' or 'dynamic'. Current value: ${VCPKG_LIBRARY_LINKAGE}")
endif()

foreach(build_suffix dest_dir IN ZIP_LISTS BUILD_SUFFIX DEST_DIR)
  vcpkg_execute_required_process(
    COMMAND make "-j8"
    WORKING_DIRECTORY "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-${build_suffix}"
    LOGNAME "build-${TARGET_TRIPLET}-${build_suffix}"
  )

  file(
    INSTALL "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-${build_suffix}/build/${LIB_FILE}"
    DESTINATION "${dest_dir}/lib"
  )

  file(GLOB CADICAL_HEADERS "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-${build_suffix}/src/*.hpp")
  file(
    INSTALL ${CADICAL_HEADERS}
    DESTINATION "${dest_dir}/include"
  )
endforeach()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")