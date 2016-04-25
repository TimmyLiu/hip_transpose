cmake_minimum_required( VERSION 2.8.12 )
project( HIP_TRANSPOSE CXX )

set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")
set(CMAKE_C_COMPILER /home/tester/Documents/github/HIP/HIP0412/HIP/bin/hipcc)
set(CMAKE_CXX_COMPILER /home/tester/Documents/github/HIP/HIP0412/HIP/bin/hipcc)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

if(TARGET_PLATFORM EQUAL 32 OR TARGET_PLATFORM EQUAL 64)
    set(TARGET_PLATFORM ${TARGET_PLATFORM} CACHE STRING "Target platform type (32-bit or 64-bit)" FORCE)
else()
    if(CMAKE_SIZEOF_VOID_P MATCHES 8)
        set(TARGET_PLATFORM "64" CACHE STRING "Target platform type (32-bit or 64-bit)" FORCE)
    else()
        set(TARGET_PLATFORM "32" CACHE STRING "Target platform type (32-bit or 64-bit)" FORCE)
    endif()
endif()

message(STATUS "Target platform: ${TARGET_PLATFORM}-bit")
if(TARGET_PLATFORM EQUAL 32)
    set(_arch "x86" INTERNAL)
    set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS FALSE)
else()
    set(_arch "x86_64" INTERNAL)
    set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)
endif()

set (HIP_TRANSPOSE_SRC
     transpose_kernel.cpp
     transpose_test.cpp)

set (HIP_TRANSPOSE_HEADER
     transpose_kernel.h)

include_directories(${CMAKE_SOURCE_DIR})

add_executable(HIP_TRANSPOSE_EXE ${HIP_TRANSPOSE_SRC})
