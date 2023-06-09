cmake_minimum_required(VERSION 3.5) 
PROJECT (my_opencv_package)

# if(CMAKE_COMPILER_IS_GNUCC)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
# endif()

find_package(OpenCV REQUIRED )
if (${OpenCV_FOUND})
  message("[MY_MESSAGE] OpenCV version: ${OpenCV_VERSION} found from the path: ${OpenCV_INCLUDE_DIRS}")
  else()
  message(FATAL_ERROR "OpenCV not found!")
endif()

find_package(PkgConfig REQUIRED)
pkg_search_module(TESSERACT REQUIRED tesseract)
pkg_search_module(LEPTONICA REQUIRED lept)
include_directories(${TESSERACT_INCLUDE_DIRS})
include_directories(${LEPTONICA_INCLUDE_DIRS})
link_directories( ${TESSERACT_LIBRARY_DIRS} )
link_directories( ${LEPTONICA_LIBRARY_DIRS} )

set( SRC_FILES_PATH_YOLO src/yolo_v5.cpp  src/helper_functions.cpp)

# Debug using VSCode: https://code.visualstudio.com/docs/cpp/cmake-linux
# message("[MY_MESSAGE] CMake reading source files from: ${SRC_FILES_PATH_IMG_DISP}")
# Ref: https://stackoverflow.com/questions/13970377/configuring-an-c-opencv-project-with-cmake
set( INCLUDE_FILES_PATH       
    #  include/header.h
    include/
)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS} )
message("[MY_MESSAGE] CMake reading include files from: ${CMAKE_CURRENT_SOURCE_DIR}/include ; ${OpenCV_INCLUDE_DIRS}. Check INCLUDE_DIRECTORIES in CMakeLists.txt to double confirm")

link_directories( ${CMAKE_BINARY_DIR}/bin)
message("[MY_MESSAGE] CMake link_directories set to: ${CMAKE_BINARY_DIR}/bin")

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

set(prcv_final_project prcv_final_project)
add_executable( ${prcv_final_project} ${SRC_FILES_PATH_YOLO} )
target_link_libraries( ${prcv_final_project} ${OpenCV_LIBS} tesseract lept ${Tesseract_LIBRARIES} ${Leptonica_LIBRARIES} )




# message("[MY_MESSAGE] Executable ${p02_write_csv} is linked against: ${OpenCV_LIBS}")
message("[MY_MESSAGE] Executable is saved to: ${EXECUTABLE_OUTPUT_PATH}/${prcv_final_project}")
