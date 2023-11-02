add_library(pargibbs
  pargibbs/lattice/lattice.cc
  pargibbs/mpi_helper.cc
  pargibbs/common/helpers.cc)

if (${USE_MPI})
  target_compile_definitions(pargibbs PUBLIC USE_MPI=true)
else()
  target_compile_definitions(pargibbs PUBLIC USE_MPI=false)
  target_include_directories(pargibbs SYSTEM PUBLIC "${CMAKE_SOURCE_DIR}/external")
endif()

# TODO: Make this more robust/ use a FindMETIS.cmake from somewhere
if (${USE_METIS})
   target_compile_definitions(pargibbs PUBLIC USE_METIS=true)

   target_include_directories(pargibbs SYSTEM PUBLIC "${METIS_ROOT}/include/")
   target_link_directories(pargibbs PUBLIC "${METIS_ROOT}/lib")
   target_link_libraries(pargibbs PUBLIC metis GKlib)
endif()

target_include_directories(pargibbs PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_compile_options(pargibbs PRIVATE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
target_link_libraries(pargibbs PUBLIC Eigen3::Eigen ${MPI_TARGET})
