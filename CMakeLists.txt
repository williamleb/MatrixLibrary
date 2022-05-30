cmake_minimum_required(VERSION 3.15)

project(GTI320_labo01)

#--------------------------------------------------
# Add googletest and setup build
#--------------------------------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# No need for GMock
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)

# Turn on folders in the solution explorer
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

FetchContent_MakeAvailable(googletest)

#--------------------------------------------------
# Define sources and executable
#--------------------------------------------------
file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB_RECURSE CMAKE ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt)
add_executable(labo01 ${SOURCES} ${HEADERS} ${CMAKE})

add_subdirectory(tests)
source_group(tests REGULAR_EXPRESSION "tests/*")

# Add linking information for Google Test
target_link_libraries(labo01 gtest)



