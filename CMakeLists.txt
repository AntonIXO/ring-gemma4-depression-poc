cmake_minimum_required(VERSION 3.16)
project(cognitive_state_system VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(cogstate
  src/csv.cpp
  src/data_fusion.cpp
  src/math_harmonization.cpp
  src/embedding.cpp
  src/fusion.cpp
  src/aim.cpp
  src/gemma4.cpp
  src/vector_db.cpp
)

add_executable(cognitive_pipeline apps/pipeline_main.cpp)
target_link_libraries(cognitive_pipeline PRIVATE cogstate)

add_executable(cognitive_train apps/train_main.cpp)
target_link_libraries(cognitive_train PRIVATE cogstate)

add_executable(cognitive_vector_db apps/vector_db_main.cpp)
target_link_libraries(cognitive_vector_db PRIVATE cogstate)

