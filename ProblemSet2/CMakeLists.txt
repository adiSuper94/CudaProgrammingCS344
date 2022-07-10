cmake_minimum_required(VERSION 3.21)
project(HW2 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

file(GLOB hdr *.hpp *.h)
file(GLOB cu *.cu)

SET(HW2_files main.cpp reference_calc.cpp compare.cpp)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CUDA_STANDARD_INCLUDE_DIRECTORIES} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_executable(HW2 ${HW2_files} ${hdr} ${cu})
set_property(TARGET HW2 PROPERTY CUDA_ARCHITECTURES 52 61 75)
target_link_libraries(HW2 ${OpenCV_LIBS})