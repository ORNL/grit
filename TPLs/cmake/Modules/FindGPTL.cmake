# module for GPTL
include(FindPackageHandleStandardArgs)

IF(NOT GPTL_DIR)
   SET(GPTL_DIR "${CMAKE_SOURCE_DIR}/../TPLs/GPTL")
ENDIF()

find_path(
    GPTL_INCLUDE_DIR gptl.h
    HINTS ${GPTL_DIR}
    PATH_SUFFIXES include
)

find_library(
    GPTL_LIBRARY
    NAMES gptl
    HINTS ${GPTL_DIR}
    PATH_SUFFIXES lib
)

find_package_handle_standard_args(
    GPTL  DEFAULT_MSG
    GPTL_LIBRARY GPTL_INCLUDE_DIR
)

IF(GPTL_FOUND)
  SET(GPTL_LIBRARIES
      ${GPTL_LIBRARY}
  )
  SET(GPTL_INCLUDE_DIRS
      ${GPTL_INCLUDE_DIR}
  )
  mark_as_advanced(GPTL_INCLUDE_DIRS GPTL_LIBRARIES )
ELSE()
  SET(GPTL_DIR "" CACHE PATH
    "An optional hint to the GPTL installation directory"
    )
ENDIF()
