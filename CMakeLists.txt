cmake_minimum_required(VERSION 3.20)
project(OctreeTransportHPC VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32 AND CMAKE_CXX_COMPILER MATCHES "ucrt64")
    get_filename_component(_MSYS2_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_MSYS2_PREFIX "${_MSYS2_BIN_DIR}" DIRECTORY)
    list(PREPEND CMAKE_PREFIX_PATH "${_MSYS2_PREFIX}")
elseif(WIN32 AND DEFINED ENV{MSYS2_PREFIX})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{MSYS2_PREFIX}")
elseif(WIN32 AND DEFINED ENV{MSYS2_ROOT})
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{MSYS2_ROOT}/ucrt64")
endif()

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTest REQUIRED)
find_path(GLM_INCLUDE_DIR glm/glm.hpp REQUIRED)

if(TARGET GLEW::GLEW)
    set(TRANSPORT_GLEW_TARGET GLEW::GLEW)
elseif(TARGET GLEW::glew)
    set(TRANSPORT_GLEW_TARGET GLEW::glew)
else()
    message(FATAL_ERROR "GLEW package did not provide an imported target")
endif()

if(TARGET glfw)
    set(TRANSPORT_GLFW_TARGET glfw)
elseif(TARGET glfw3::glfw)
    set(TRANSPORT_GLFW_TARGET glfw3::glfw)
else()
    message(FATAL_ERROR "glfw3 package did not provide an imported target")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3)
elseif(MSVC)
    add_compile_options(/O2)
endif()

set(LIB_SOURCES
    src/cli_config.cpp
    src/phase_space_seeder.cpp
    src/octree_arena.cpp
    src/solvers_cpu.cpp
    src/gpu_compute_engine.cpp
    src/gl_renderer.cpp
    src/physics_diagnostics.cpp
)

add_library(transport_lib STATIC ${LIB_SOURCES})
target_include_directories(transport_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GLM_INCLUDE_DIR}
)
target_link_libraries(transport_lib PUBLIC
    OpenMP::OpenMP_CXX
    ${TRANSPORT_GLEW_TARGET}
    ${TRANSPORT_GLFW_TARGET}
)
if(WIN32)
    target_link_libraries(transport_lib PUBLIC opengl32 gdi32)
else()
    target_link_libraries(transport_lib PUBLIC OpenGL::GL)
endif()

add_executable(octree_transport_hpc src/main.cpp)
target_link_libraries(octree_transport_hpc PRIVATE transport_lib)
add_custom_command(TARGET octree_transport_hpc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        $<TARGET_FILE_DIR:octree_transport_hpc>/shaders
)

enable_testing()
add_executable(octree_transport_tests tests/test_octree_transport.cpp)
target_link_libraries(octree_transport_tests PRIVATE
    transport_lib
    GTest::gtest
    GTest::gtest_main
)
add_custom_command(TARGET octree_transport_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        $<TARGET_FILE_DIR:octree_transport_tests>/shaders
)
add_test(NAME OctreeTransportTests COMMAND octree_transport_tests)
