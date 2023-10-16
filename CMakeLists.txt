cmake_minimum_required(VERSION 3.9)

project(sscma)

find_package(OpenCV REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

option(USE_SYSTEM_NCNN "build with system libncnn" OFF)
#option(USE_SYSTEM_YAML-CPP "build with system libyaml-cpp" OFF)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ncnn.cmake)
#include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/yaml-cpp.cmake)

macro(add_example name)
    add_executable(${name} ${CMAKE_CURRENT_SOURCE_DIR}/examples/${name}.cpp)
    target_link_libraries(${name} PRIVATE ncnn  ${OpenCV_LIBS} )
    target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/examples ${CMAKE_CURRENT_SOURCE_DIR}/src)
endmacro()


add_example(sscma-yolov8)
add_example(sscma-yolov5)

