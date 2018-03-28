#module for silo
include(FindPackageHandleStandardArgs)

if(NOT SILO_DIR)
  SET(SILO_DIR "${CMAKE_SOURCE_DIR}/../TPLs/silo")
endif()

find_path(
  SILO_INCLUDE_DIR silo.h
  HINTS ${SILO_DIR}
  PATH_SUFFIXES include
)

find_library(
  SILO_LIBRARY
  NAMES silo siloh5
  HINTS ${SILO_DIR}
  PATH_SUFFIXES lib
)

find_package_handle_standard_args(
    SILO  DEFAULT_MSG
    SILO_LIBRARY SILO_INCLUDE_DIR
)

IF(SILO_FOUND)
  SET(SILO_LIBRARIES
      ${SILO_LIBRARY}
  )
  SET(SILO_INCLUDE_DIRS
      ${SILO_INCLUDE_DIR}
  )
  mark_as_advanced(SILO_INCLUDE_DIRS SILO_LIBRARIES )
ELSE()
  SET(SILO_DIR "" CACHE PATH
    "An optional hint to the SILO installation directory"
    )
ENDIF()
