cmake_minimum_required(VERSION 3.9)

project(sscma)

if(NOT DEFINED APP)
    message(FATAL_ERROR "No APP specified")
endif()
find_package(OpenCV REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release CACHE STRING "Choose the type of build" FORCE)
endif()

option(USE_SYSTEM_NCNN "build with system libncnn" OFF)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ncnn.cmake)

include(${CMAKE_CURRENT_SOURCE_DIR}/examples/${APP}/app.cmake)



