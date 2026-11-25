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


# CPU KV-cache update primitive (no QNN dependency).
add_library(gemma4-kv-cache STATIC src/kv_cache.cpp)
target_include_directories(gemma4-kv-cache PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_compile_options(gemma4-kv-cache PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(gemma4-qnn-run PRIVATE gemma4-kv-cache)
if(BUILD_TESTING)
  add_executable(gemma4-kv-cache-test tests/kv_cache_test.cpp)
  target_link_libraries(gemma4-kv-cache-test PRIVATE gemma4-kv-cache)
  add_test(NAME kv_cache_slot_layout COMMAND gemma4-kv-cache-test)
endif()

# Stage 5: check installed native TFLite C ABI without TensorFlow dev headers.
add_executable(gemma4-cpu-abi src/cpu_abi_probe.cpp)
target_link_libraries(gemma4-cpu-abi PRIVATE dl)
target_compile_options(gemma4-cpu-abi PRIVATE -Wall -Wextra -Wpedantic)

# Stage 5: named CPU TFLite signature runner using validated native C ABI.
add_executable(gemma4-cpu-run src/cpu_runner.cpp)
target_link_libraries(gemma4-cpu-run PRIVATE dl)
target_compile_options(gemma4-cpu-run PRIVATE -Wall -Wextra -Wpedantic)
if(BUILD_TESTING)
  add_test(NAME cpu_runner_help COMMAND gemma4-cpu-run --help)
endif()
