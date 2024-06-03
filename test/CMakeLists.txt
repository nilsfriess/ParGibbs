cmake_minimum_required(VERSION 3.20)
project(ParMGMC_Tests
	LANGUAGES CXX)

file(DOWNLOAD
    https://github.com/doctest/doctest/releases/download/v2.4.11/doctest.h
    ${CMAKE_CURRENT_BINARY_DIR}/external/doctest.h)

find_package(PkgConfig REQUIRED)
pkg_search_module(parmgmc parmgmc>=0.0.1 IMPORTED_TARGET REQUIRED)
pkg_search_module(PETSc PETSc>=3.19 IMPORTED_TARGET REQUIRED)

set(PARMGMC_CC "-I${CMAKE_CURRENT_BINARY_DIR}/external ${parmgmc_CFLAGS} ${PETSc_CFLAGS}")
set(PARMGMC_CC "${PARMGMC_CC} -Wl,-rpath,${parmgmc_LIBRARY_DIRS},-rpath,${PETSc_LIBRARY_DIRS}")
set(PARMGMC_CC "${PARMGMC_CC} -L${parmgmc_LIBRARY_DIRS} -L${PETSc_LIBRARY_DIRS}")
set(PARMGMC_CC "${PARMGMC_CC} -l${parmgmc_LIBRARIES} -l${PETSc_LIBRARIES} -lmpi")
string(REPLACE ";" " " PARMGMC_COMP "${PARMGMC_CC}")
message(STATUS "LIT compile arguments: ${PARMGMC_COMP}")

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)
add_custom_target(check
		  COMMAND lit "${CMAKE_CURRENT_BINARY_DIR}" -v)
