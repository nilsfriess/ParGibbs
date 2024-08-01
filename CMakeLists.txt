cmake_minimum_required(VERSION 3.20)
project(ParMGMC
	LANGUAGES C CXX
	VERSION 0.0.1
	DESCRIPTION "Implementation of the Multigrid Monte Carlo method in PETSc"
	HOMEPAGE_URL "https://github.com/nilsfriess/ParMGMC.git")

set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# find_package(MPI REQUIRED)
# set(MPI_TARGET MPI::MPI_C)

find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC PETSc>=3.17 IMPORTED_TARGET REQUIRED)

if (${PETSC_FOUND})
  message(STATUS "Found PETSc version ${PETSC_VERSION}, include dirs: ${PETSC_INCLUDE_DIRS}")
endif()

add_library(parmgmc SHARED)
file(GLOB_RECURSE PARMGMC_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include/parmgmc/*.h")
message(STATUS ${PARMGMC_INCLUDES})
target_sources(parmgmc
	PRIVATE
	    src/pc_hogwild.c
	    src/pc_gibbs.c
	    src/pc_gamgmc.c
	    src/pc_chols.c
	    src/ziggurat.c
	    src/mc_sor.c
	    src/parmgmc.c
	    src/problems.c
	    src/ms.c
    	    src/cgs.c
	    src/obs.c
	 PUBLIC
	    FILE_SET parmgmcheaders
	    BASE_DIRS include
	    TYPE HEADERS
	       FILES ${PARMGMC_INCLUDES})
# target_include_directories(parmgmc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(parmgmc SYSTEM PUBLIC ${PETSC_INCLUDE_DIRS})
target_compile_options(parmgmc PRIVATE -Wall -Wextra -Wpedantic -pedantic -Wshadow -Wpointer-arith -Wcast-qual -Wstrict-prototypes -Wmissing-prototypes)
target_compile_options(parmgmc PUBLIC ${PETSC_CFLAGS})
target_link_libraries(parmgmc PUBLIC PkgConfig::PETSC)

# add_executable(test test.c)
# target_include_directories(test PRIVATE include)
# target_compile_options(test PRIVATE -Wall -Wextra -Wpedantic)
# target_link_libraries(test parmgmc)

include(GNUInstallDirs)
list(APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_LIBDIR})
install(TARGETS parmgmc 
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	FILE_SET parmgmcheaders
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/share/pkgconfig/parmgmc.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/parmgmc.pc
    @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/parmgmc.pc
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

if(NOT DEFINED PARMGMC_ENABLE_PYTHON_BINDINGS)
  set(PARMGMC_ENABLE_PYTHON_BINDINGS False)
endif()

if(${PARMGMC_ENABLE_PYTHON_BINDINGS})
  add_subdirectory(python)
endif()

if(NOT DEFINED PARMGMC_BUILD_DOCS)
  set(PARMGMC_BUILD_DOCS False)
endif()

if(${PARMGMC_BUILD_DOCS})
  add_subdirectory(docs)
endif()

