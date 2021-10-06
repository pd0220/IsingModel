cmake_minimum_required(VERSION 3.0.0)
project(IsingClassic VERSION 0.1.0 LANGUAGES CXX CUDA)

if (MSVC)
  string(REGEX REPLACE "/W[0-9]" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif (MSVC)


include(CTest)
enable_testing()

add_executable(IsingClassic IsingClassic.cpp)
add_executable(IsingCPU IsingCPU.cpp)
add_executable(IsingGPU IsingGPU.cu)
add_executable(RENORM_HW RENORM_HW.cu)
    
set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17
                                                 CXX_STANDARD_REQUIRED ON
                                                 CXX_EXTENSIONS OFF)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>
                                               $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->)