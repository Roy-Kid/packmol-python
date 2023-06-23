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
set(packmol_src_path "${CMAKE_SOURCE_DIR}/external/packmol/src")
set(patch_src_path "${CMAKE_SOURCE_DIR}/external")
# Glob all f90 file under external/packmol/src
# file(GLOB_RECURSE fortran_src_files "${CMAKE_SOURCE_DIR}/external/packmol/src/*.f90")

set(fortran_src_files
    ${packmol_src_path}/sizes.f90
    # ${packmol_src_path}/ahestetic.f90
    # ${packmol_src_path}/cenmass.f90
    # ${packmol_src_path}/checkpoint.f90
    # ${packmol_src_path}/comparegrad.f90
    # ${packmol_src_path}/comprest.f90
    ${packmol_src_path}/compute_data.f90
    # ${packmol_src_path}/computef.f90
    # ${packmol_src_path}/computeg.f90
    # ${packmol_src_path}/exit_codes.f90
    ${packmol_src_path}/flashmod.f90
    # ${packmol_src_path}/fparc.f90
    # ${packmol_src_path}/gencan_ieee_signal_routines.f90
    # ${packmol_src_path}/gencan.f
    # # ${packmol_src_path}/getinp.f90
    # ${packmol_src_path}/gparc.f90
    # ${packmol_src_path}/gwalls.f90
    # ${packmol_src_path}/heuristics.f90
    # ${packmol_src_path}/initial.f90
    ${packmol_src_path}/input.f90
    # ${packmol_src_path}/jacobi.f90
    # ${packmol_src_path}/output.f90
    # ${packmol_src_path}/pgencan.f90
    # ${packmol_src_path}/polartocart.f90
    # ${packmol_src_path}/random.f90
    # ${packmol_src_path}/resetboxes.f90
    # ${packmol_src_path}/setibox.f90
    # ${packmol_src_path}/setijk.f90
    # ${packmol_src_path}/setsizes.f90
    # ${packmol_src_path}/strlength.f90
    # ${packmol_src_path}/swaptypemod.f90
    # ${packmol_src_path}/title.f90
    # ${packmol_src_path}/tobar.f90
    ${packmol_src_path}/usegencan.f90
    # ${packmol_src_path}/writesuccess.f90
    ${packmol_src_path}/flashsort.f90

    ${patch_src_path}/set_params.f90
)

set(f2py_module_c "${f2py_module_name}module.c")

# Generate sources
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}
    COMMAND ${Python_EXECUTABLE} -m "numpy.f2py"
    -m ${f2py_module_name}
    ${fortran_src_files}
    --lower
    DEPENDS ${fortran_src_files}
    )
    
add_custom_target(
    packmol_lib
    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c}"
)

# Set up target
python_add_library(${f2py_module_name} MODULE WITH_SOABI
    ${CMAKE_CURRENT_BINARY_DIR}/${f2py_module_c} # Generated
    ${F2PY_INCLUDE_DIR}/fortranobject.c # From Numpy
    ${fortran_src_files} # From Packmol
)

# Depend on sources
target_link_libraries(${f2py_module_name} PRIVATE Python::NumPy)
add_dependencies(${f2py_module_name} packmol_lib)
target_include_directories(${f2py_module_name} PRIVATE "${F2PY_INCLUDE_DIR}")
