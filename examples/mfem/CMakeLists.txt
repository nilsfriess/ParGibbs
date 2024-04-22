include(FetchContent)
FetchContent_Declare(
  pcgcpp
  GIT_REPOSITORY https://github.com/imneme/pcg-cpp.git
)
FetchContent_MakeAvailable(pcgcpp)
cmake_path(SET pcgcpp_INCLDUE_DIR "${pcgcpp_SOURCE_DIR}/include")

if(${mfem_FOUND})
  add_executable(mfem_example main.cc)

  target_include_directories(mfem_example SYSTEM PRIVATE ${pcgcpp_INCLDUE_DIR})
  target_compile_options(mfem_example PRIVATE -Wall -Wextra -Wpedantic -Wnon-virtual-dtor)
  target_link_libraries(mfem_example parmgmc mfem)
endif()
