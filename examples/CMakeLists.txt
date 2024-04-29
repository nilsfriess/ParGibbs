include(FetchContent)
FetchContent_Declare(
  pcgcpp
  GIT_REPOSITORY https://github.com/imneme/pcg-cpp.git
)
FetchContent_MakeAvailable(pcgcpp)
cmake_path(SET pcgcpp_INCLDUE_DIR "${pcgcpp_SOURCE_DIR}/include")

set(MFEM_DIR "" CACHE PATH "Path to the MFEM build or install prefix.")
find_package(mfem
             NAMES MFEM
             HINTS "${MFEM_DIR}" "${MFEM_DIR}/lib/cmake/mfem"
             NO_DEFAULT_PATH)

add_subdirectory(general)
add_subdirectory(statistics)
add_subdirectory(mfem)
add_subdirectory(scaling)
