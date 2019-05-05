CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

SET(BUILD_TYPE Release)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

PROJECT(gn_ba)
set(Sophus_DIR /home/sunmb/lib/Sophus/build)
find_package(Sophus REQUIRED)

MESSAGE(STATUS "Sophus INCLUDE: ${Sophus_INCLUDE_DIRS} \n LIBS: ${Sophus_LIBRARIES}")

INCLUDE_DIRECTORIES("/usr/include/eigen3" "/usr/local/include/sophus")

ADD_EXECUTABLE(${PROJECT_NAME} GN_BA.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${Sophus_LIBRARIES})
