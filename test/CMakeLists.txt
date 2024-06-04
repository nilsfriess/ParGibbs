cmake_minimum_required(VERSION 3.20)
project(ParMGMC_Tests
	LANGUAGES C)

find_package(PkgConfig REQUIRED)
pkg_search_module(parmgmc parmgmc>=0.0.1 REQUIRED)
pkg_search_module(PETSc PETSc>=3.19 REQUIRED)
pkg_search_module(MPICH mpich REQUIRED)

pkg_get_variable(MPIRUN_PATH ${MPICH_MODULE_NAME} prefix)
set(MPIRUN_PATH "${MPIRUN_PATH}/bin/mpirun")
message(STATUS "mpirun path: ${MPIRUN_PATH}")

set(PARMGMC_CC "${parmgmc_CFLAGS} ${PETSc_CFLAGS}")
set(PARMGMC_CC "${PARMGMC_CC} -Wl,-rpath,${parmgmc_LIBRARY_DIRS},-rpath,${PETSc_LIBRARY_DIRS}")
set(PARMGMC_CC "${PARMGMC_CC} -L${parmgmc_LIBRARY_DIRS} -L${PETSc_LIBRARY_DIRS} -L${MPICH_LIBRARY_DIRS}")
set(PARMGMC_CC "${PARMGMC_CC} -l${parmgmc_LIBRARIES} -l${PETSc_LIBRARIES} -lmpi")
string(REPLACE ";" " " PARMGMC_COMP "${PARMGMC_CC}")
set(PARMGMC_COMP "${PARMGMC_COMP} ${CMAKE_C_FLAGS}")
message(STATUS "LIT compile arguments: ${PARMGMC_COMP}")

configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)
add_custom_target(check-seq
		  COMMAND lit "${CMAKE_CURRENT_BINARY_DIR}/tests" -v -DNP=1)
add_custom_target(check-par
		  COMMAND lit "${CMAKE_CURRENT_BINARY_DIR}/tests" -v -DNP=4)
