cmake_minimum_required(VERSION 3.21)
project(ProblemSet1 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu)

SET (HW1_files main.cpp reference_calc.cpp compare.cpp)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} ${CMAKE_CUDA_STANDARD_INCLUDE_DIRECTORIES} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_library(student_func STATIC ${HW1_files} compare.h reference_calc.h HW1.cpp utils.h timer.h student_func.cu)
set_target_properties(student_func PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_executable(ProblemSet1 ${HW1_files} ${hdr} ${cu})
target_link_libraries(ProblemSet1 ${OpenCV_LIBS})