cmake_minimum_required(VERSION 3.19)

project(packmol-python
    VERSION 1.0.0
    DESCRIPTION "Python bindings for Packmol"
    LANGUAGES C Fortran
)

# Safety net
if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(FATAL_ERROR "In-source builds are not allowed. Please use a separate directory for build files.")
endif()

# Grab python, 3.8 or newer
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# Grab the variables from a local Python installation
# F2PY headers
execute_process(
  COMMAND "${Python_EXECUTABLE}"
  -c "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Print out the discovered paths
include(CMakePrintHelpers)
cmake_print_variables(Python_INCLUDE_DIRS)
cmake_print_variables(F2PY_INCLUDE_DIR)
cmake_print_variables(Python_NumPy_INCLUDE_DIRS)

# Common variables
set(f2py_module_name "packmol")
# Glob all f90 file under external/packmol/src
set(packmol_src_path "${CMAKE_SOURCE_DIR}/external/packmol/src")
file(GLOB_RECURSE fortran_src_files "${CMAKE_SOURCE_DIR}/external/packmol/src/*.f90")
set(f2py_module_c "${f2py_module_name}module.c")

# Generate sources
add_custom_target(
    packmol_lib
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}"
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}
    COMMAND ${Python_EXECUTABLE} -m "numpy.f2py"
        ${fortran_src_files}
        -m ${f2py_module_name}
        --lower
    DEPENDS ${fortran_src_files}
)

# Set up target
python_add_library(packmol_python MODULE WITH_SOABI
    "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}" # Generated
    "${F2PY_INCLUDE_DIR}/fortranobject.c" # From Numpy
    ${fortran_src_files} # From Packmol
)

# Depend on sources
target_link_libraries(packmol_python PRIVATE Python::NumPy)
add_dependencies(packmol_python packmol_lib)
target_include_directories(packmol_python PRIVATE "${F2PY_INCLUDE_DIR}")
