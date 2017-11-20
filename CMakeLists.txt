cmake_minimum_required(VERSION 2.6)
# set(OpenCV_DIR "/usr/local/opencv-3.3.0/share/OpenCV")
# include_directories("/usr/local/opencv-3.3.0/include")
project(ml)
include_directories(include)
include_directories(3rdparty)
file(GLOB headers include/andres/* include/andres/ml/* 3rdparty/GLRenderer/include/* 3rdparty/POSEST/include/*  3rdparty/POSEST/include/levmar/* 3rdparty/POSEST/include/mlsl/* include/edge/* include/tools/* include/ObjectDetector/*)
enable_testing()

aux_source_directory(./src/  SRC_LIST)
aux_source_directory(./3rdparty/GLRenderer/src/ SRC_LIST)
# aux_source_directory(./3rdparty/View/ SRC_LIST)
aux_source_directory(./3rdparty/POSEST/src/ SRC_LIST)
aux_source_directory(./3rdparty/POSEST/src/mlsl/ SRC_LIST)
aux_source_directory(./3rdparty/POSEST/src/EdgeDistanceFieldTraking.cpp SRC_LIST)

aux_source_directory(./3rdparty/POSEST/src/levmar/Axb.c SRC_LIST)
aux_source_directory(./3rdparty/POSEST/src/levmar/expfit.c SRC_LIST)
aux_source_directory(./3rdparty/POSEST/src/levmar/lmbc.c SRC_LIST)
aux_source_directory(./3rdparty/POSEST/src/levmar/lmblec.c SRC_LIST)
aux_source_directory(./3rdparty/POSEST/src/levmar/lmbleic.c SRC_LIST)
aux_source_directory(./3rdparty/POSEST/src/levmar/mestimators.c SRC_LIST)
aux_source_directory(./3rdparty/POSEST/src/levmar/misc.c SRC_LIST)

# aux_source_directory(./src/DT.cpp  SRC_LIST)

set( CMAKE_CXX_COMPILER "g++" )
set( CMAKE_BUILD_TYPE "Debug" )

# list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
# set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
# set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )



############################################################################## 
##glog
##############################################################################
find_package(glog 0.3.5 REQUIRED)
include_directories(BEFORE ${GLOG_INCLUDE_DIRS}) 

##############################################################################
# Doxygen
##############################################################################
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
    message(STATUS "Doxygen found")
else()
    message("doxygen not found")
endif()

##############################################################################
# C++11 support
##############################################################################
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  set( CMAKE_CXX_FLAGS "-std=c++14 -march=native -O3" )
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Some functionality will not be available.")
endif()

##############################################################################
# MSVC-specific settings
##############################################################################
if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

##############################################################################
# OpenMP
##############################################################################
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMP found")
else()
    message("OpenMP not found")
endif()

##############################################################################
# OpenGL
##############################################################################

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )
# target_link_libraries(testas  )

# Sophus 
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )
##############################################################################
# Eigen
##############################################################################
include_directories( "/usr/include/eigen3" )
##############################################################################
# OpenCV
##############################################################################
find_package( OpenCV 3.3 REQUIRED )
# if(OPENCV_FOUND)
include_directories( ${OpenCV_INCLUDE_DIRS} )
#     message(STATUS "OpenCV found")
# else()d
#     message("OpenCV not found")
# endif()

##############################################################################
# targets
##############################################################################
# 
add_executable(test-obj-detector src/unittest/test-obj-detector.cpp ${headers} ${SRC_LIST})
add_test(test-obj-detector test-obj-detector)
target_link_libraries( test-obj-detector  ${OpenCV_LIBS}  ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLOG_LIBRARY}   ${Sophus_LIBRARIES} glog::glog   -O2 blas lapack
) # exe 链接 OpenCV


# add_executable(test-DT src/unittest/test-DT.cpp ${headers} ${SRC_LIST})
# add_test(test-DT test-Edge-detector)
# target_link_libraries( test-DT ${OpenCV_LIBS}  ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLOG_LIBRARY} glog::glog) # exe 链接 OpenCV

add_executable(test-model src/unittest/test-model.cpp ${headers} ${SRC_LIST})
add_test(test-model test-Edge-detector)
target_link_libraries( test-model ${OpenCV_LIBS}  ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLOG_LIBRARY}     ${Sophus_LIBRARIES} glog::glog     -O2 blas lapack
) # exe 链接 OpenCV



add_executable(test-optimizer src/unittest/test-optimizer.cpp ${headers} ${SRC_LIST})
add_test(test-optimizer test-Edge-detector)
target_link_libraries( test-optimizer ${OpenCV_LIBS}  ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLOG_LIBRARY}  ${Sophus_LIBRARIES} glog::glog     -O2 blas lapack
) # exe 链接 OpenCV

# set( 
#     THIRD_PARTY_LIBS
#     ${OpenCV_LIBS}
# )
# if(DOXYGEN_FOUND)
#     configure_file("${ml_SOURCE_DIR}/doxygen/doxyfile-ml-decision-trees.in" "${ml_BINARY_DIR}/doxyfile-ml-decision-trees" @ONLY IMMEDIATE)
#     add_custom_target(doc-ml-decision-trees ALL COMMAND ${DOXYGEN} "${ml_BINARY_DIR}/doxyfile-ml-decision-trees")
# endif()

