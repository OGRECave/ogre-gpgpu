cmake_minimum_required(VERSION 3.10)
project(OgreCuda CXX CUDA)

find_package(OGRE 1.12 REQUIRED)

file(COPY ${OGRE_CONFIG_DIR}/plugins.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(WRITE ${CMAKE_BINARY_DIR}/resources.cfg  "[General]\nFileSystem=${PROJECT_SOURCE_DIR}/media/skybox\n")

set(LIB_SRCS OgreCuda/src/OgreCuda.cpp OgreCuda/src/OgreCudaGL.cpp)
file(GLOB DEMO_SRCS OgreCudaDemo/src/*)

add_executable(OgreCudaDemo ${LIB_SRCS} ${DEMO_SRCS})

target_link_libraries(OgreCudaDemo OgreMain cudart OIS)
target_include_directories(OgreCudaDemo PRIVATE OgreCuda/include OgreCudaDemo/include ${OGRE_INCLUDE_DIRS}/RenderSystems/GL)
target_compile_definitions(OgreCudaDemo PRIVATE OGRE_CUDA_LIB)