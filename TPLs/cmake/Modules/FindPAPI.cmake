# module for PAPI
include(FindPackageHandleStandardArgs)

IF(NOT PAPI_DIR)
   SET(PAPI_DIR "${CMAKE_SOURCE_DIR}/../TPLs/PAPI")
ENDIF()

find_path(
    PAPI_INCLUDE_DIR papi.h
    HINTS ${PAPI_DIR}
    PATH_SUFFIXES include
)

find_library(
    PAPI_LIBRARY
    NAMES papi
    HINTS ${PAPI_DIR}
    PATH_SUFFIXES lib
)

find_package_handle_standard_args(
    PAPI  DEFAULT_MSG
    PAPI_LIBRARY PAPI_INCLUDE_DIR
)

IF(PAPI_FOUND)
  SET(PAPI_LIBRARIES
      ${PAPI_LIBRARY}
  )
  SET(PAPI_INCLUDE_DIRS
      ${PAPI_INCLUDE_DIR}
  )
  mark_as_advanced(PAPI_INCLUDE_DIRS PAPI_LIBRARIES )
ELSE()
  SET(PAPI_DIR "" CACHE PATH
    "An optional hint to the PAPI installation directory"
    )
ENDIF()
