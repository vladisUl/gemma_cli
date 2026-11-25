cmake_minimum_required(VERSION 3.16)
project(gemma4_qnn_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED QAIRT_ROOT)
  message(FATAL_ERROR "Set -DQAIRT_ROOT=/path/to/QAIRT/2.47.0.260601")
endif()
if(NOT EXISTS "${QAIRT_ROOT}/include/QNN/QnnInterface.h")
  message(FATAL_ERROR "QAIRT headers not found: ${QAIRT_ROOT}/include/QNN")
endif()
add_executable(gemma4-qnn-probe src/qnn_probe.cpp)
target_include_directories(gemma4-qnn-probe PRIVATE "${QAIRT_ROOT}/include")
target_link_libraries(gemma4-qnn-probe PRIVATE dl)
target_compile_options(gemma4-qnn-probe PRIVATE -Wall -Wextra -Wpedantic)

# Stage 2: QNN System metadata and raw single-graph execution.
add_executable(gemma4-qnn-run src/qnn_runner.cpp)
# QAIRT System headers use includes such as "System/QnnSystemCommon.h".
# The copied SDK layout keeps QNN under include/QNN, so we need both roots.
target_include_directories(gemma4-qnn-run PRIVATE
  "${QAIRT_ROOT}/include"
  "${QAIRT_ROOT}/include/QNN"
)
target_link_libraries(gemma4-qnn-run PRIVATE dl)
target_compile_options(gemma4-qnn-run PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME qnn_runner_help COMMAND gemma4-qnn-run --help)
endif()
