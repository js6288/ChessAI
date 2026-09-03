cmake_minimum_required(VERSION 3.20)
project(chessai_native LANGUAGES CXX)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_chessai_native src/bindings.cpp)
target_compile_features(_chessai_native PRIVATE cxx_std_20)
target_compile_definitions(_chessai_native PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)

if(MSVC)
  target_compile_options(_chessai_native PRIVATE /W4 /permissive-)
else()
  target_compile_options(_chessai_native PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(MINGW)
  target_compile_options(_chessai_native PRIVATE -mthreads)
  target_link_options(_chessai_native PRIVATE -mthreads)
endif()

# Editable Python installs put ``src`` on sys.path, so the optional top-level
# extension can be imported without copying binaries into the repository.
set(CHESSAI_PYTHON_OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/../src")
set_target_properties(
  _chessai_native
  PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CHESSAI_PYTHON_OUTPUT}"
    RUNTIME_OUTPUT_DIRECTORY "${CHESSAI_PYTHON_OUTPUT}"
)

foreach(CONFIG Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${CONFIG}" CONFIG_UPPER)
  set_target_properties(
    _chessai_native
    PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CHESSAI_PYTHON_OUTPUT}"
      RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${CHESSAI_PYTHON_OUTPUT}"
  )
endforeach()
