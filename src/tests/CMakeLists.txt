find_package(Criterion)

if(CRITERION_FOUND)
    # add_executable(test_init test_init.c)
    # set_property(TARGET test_init PROPERTY C_STANDARD 99)
    # target_include_directories(test_init PRIVATE ${CRITERION_INCLUDE_DIR} ${CMAKE_SOURCE_DIR})
    # target_link_libraries(test_init PRIVATE ${CRITERION_LIBRARY} meraxes_lib)
    # add_test(NAME test_init COMMAND test_init)

    add_executable(test_parse_snaplist test_parse_snaplist.c)
    set_property(TARGET test_parse_snaplist PROPERTY C_STANDARD 99)
    target_include_directories(test_parse_snaplist PRIVATE ${CRITERION_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
    target_link_libraries(test_parse_snaplist PRIVATE ${CRITERION_LIBRARY} meraxes_lib)
    add_test(NAME test_parse_snaplist COMMAND test_parse_snaplist)

    add_executable(test_cooling test_cooling.c)
    set_property(TARGET test_cooling PROPERTY C_STANDARD 99)
    target_include_directories(test_cooling PRIVATE ${CRITERION_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
    target_link_libraries(test_cooling PRIVATE ${CRITERION_LIBRARY} meraxes_lib)
    target_compile_definitions(test_cooling PRIVATE COOLING_FUNCS_DIR="${CMAKE_SOURCE_DIR}/input/cooling_functions")
    add_test(NAME test_cooling COMMAND test_cooling)

    add_executable(test_blackhole_feedback test_blackhole_feedback.c)
    set_property(TARGET test_blackhole_feedback PROPERTY C_STANDARD 99)
    target_include_directories(test_blackhole_feedback PRIVATE ${CRITERION_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
    target_link_libraries(test_blackhole_feedback PRIVATE ${CRITERION_LIBRARY} meraxes_lib)
    add_test(NAME test_blackhole_feedback COMMAND test_blackhole_feedback)

    add_executable(test_emission_lines test_emission_lines.c)
    set_property(TARGET test_emission_lines PROPERTY C_STANDARD 99)
    target_include_directories(test_emission_lines PRIVATE ${CRITERION_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
    target_link_libraries(test_emission_lines PRIVATE ${CRITERION_LIBRARY} meraxes_lib)
    add_test(NAME test_emission_lines COMMAND test_emission_lines)
else()
    message(WARNING "Failed to find Criterion. You will not be able to run tests.")
endif(CRITERION_FOUND)

# Standalone test driver for cooling module (no Criterion required)
add_executable(main_cooling main_cooling.c)
set_property(TARGET main_cooling PROPERTY C_STANDARD 99)
target_include_directories(main_cooling PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(main_cooling PRIVATE meraxes_lib)
target_compile_definitions(main_cooling PRIVATE COOLING_FUNCS_DIR="${CMAKE_SOURCE_DIR}/input/cooling_functions")

# Standalone test driver for blackhole feedback module (no Criterion required)
add_executable(main_blackhole_feedback main_blackhole_feedback.c)
set_property(TARGET main_blackhole_feedback PROPERTY C_STANDARD 99)
target_include_directories(main_blackhole_feedback PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(main_blackhole_feedback PRIVATE meraxes_lib)

# Photon-allocation tests are ordinary executables, independent of Criterion,
# MPI execution, external simulation data and Meraxes global state.
foreach(pc_test photon_conservation photon_conservation_es photon_conservation_shape)
    add_executable(test_${pc_test}
        test_${pc_test}.c
        ${CMAKE_SOURCE_DIR}/src/physics/photon_conservation.c
        ${CMAKE_SOURCE_DIR}/src/physics/photon_conservation_geometry.c
        ${CMAKE_SOURCE_DIR}/src/physics/photon_conservation_es.c
        ${CMAKE_SOURCE_DIR}/src/physics/photon_conservation_shape.c)
    set_property(TARGET test_${pc_test} PROPERTY C_STANDARD 99)
    target_include_directories(test_${pc_test} PRIVATE ${CMAKE_SOURCE_DIR}/src/physics)
    target_link_libraries(test_${pc_test} PRIVATE m)
    # Several independent oracles use assert; keep their checks active in Release.
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_${pc_test} PRIVATE -UNDEBUG)
    endif()
    target_compile_options(test_${pc_test} PRIVATE ${SANITIZE_FLAGS})
    target_link_options(test_${pc_test} PRIVATE ${SANITIZE_FLAGS})
    add_test(NAME test_${pc_test} COMMAND test_${pc_test})
endforeach()

# Native H/He kernels: independent atom, photon, charge and energy budgets.
foreach(hhe_test photon_hhe photon_hhe_transport)
    add_executable(test_${hhe_test}
        test_${hhe_test}.c
        ${CMAKE_SOURCE_DIR}/src/physics/photon_hhe_chemistry.c
        ${CMAKE_SOURCE_DIR}/src/physics/photon_hhe_transport.c)
    set_property(TARGET test_${hhe_test} PROPERTY C_STANDARD 99)
    target_include_directories(test_${hhe_test} PRIVATE ${CMAKE_SOURCE_DIR}/src/physics)
    target_link_libraries(test_${hhe_test} PRIVATE m)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_${hhe_test} PRIVATE -UNDEBUG -fno-fast-math -ffp-contract=off)
    endif()
    target_compile_options(test_${hhe_test} PRIVATE ${SANITIZE_FLAGS})
    target_link_options(test_${hhe_test} PRIVATE ${SANITIZE_FLAGS})
    add_test(NAME test_${hhe_test} COMMAND test_${hhe_test})
endforeach()

# Complete MPI packet phase exchange, including ranks with no mesh slab.
foreach(hhe_mpi_test photon_hhe_mpi_empty photon_hhe_mpi_coupled)
    add_executable(test_${hhe_mpi_test}
        test_${hhe_mpi_test}.c
        ${CMAKE_SOURCE_DIR}/src/core/photon_hhe_mpi.c
        ${CMAKE_SOURCE_DIR}/src/physics/photon_hhe_chemistry.c
        ${CMAKE_SOURCE_DIR}/src/physics/photon_hhe_transport.c)
    set_property(TARGET test_${hhe_mpi_test} PROPERTY C_STANDARD 99)
    target_include_directories(test_${hhe_mpi_test} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_${hhe_mpi_test} PRIVATE MPI::MPI_C m)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_${hhe_mpi_test} PRIVATE -UNDEBUG -fno-fast-math -ffp-contract=off)
    endif()
    target_compile_options(test_${hhe_mpi_test} PRIVATE ${SANITIZE_FLAGS})
    target_link_options(test_${hhe_mpi_test} PRIVATE ${SANITIZE_FLAGS})
    foreach(hhe_ranks 1 2 4)
        add_test(NAME test_${hhe_mpi_test}_${hhe_ranks}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${hhe_ranks} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:test_${hhe_mpi_test}> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(test_${hhe_mpi_test}_${hhe_ranks} PROPERTIES
            PROCESSORS ${hhe_ranks} TIMEOUT 120 ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1")
    endforeach()
endforeach()

# Production adapter with prescribed snapshots and independent HDF5 audits.
add_executable(test_photon_hhe_adapter test_photon_hhe_adapter.c)
set_property(TARGET test_photon_hhe_adapter PROPERTY C_STANDARD 99)
target_include_directories(test_photon_hhe_adapter PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(test_photon_hhe_adapter PRIVATE meraxes_lib)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_photon_hhe_adapter PRIVATE -UNDEBUG -fno-fast-math -ffp-contract=off)
endif()
foreach(hhe_ranks 1 2 4)
    add_test(NAME test_photon_hhe_adapter_${hhe_ranks}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${hhe_ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:test_photon_hhe_adapter> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_photon_hhe_adapter_${hhe_ranks} PROPERTIES
        PROCESSORS ${hhe_ranks} TIMEOUT 120 ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1")
endforeach()

# Exercise arc handles above the 32-bit range without allocating billions of
# edges. This white-box fixture includes the allocator implementation itself.
add_executable(test_photon_conservation_index
    test_photon_conservation_index.c
    ${CMAKE_SOURCE_DIR}/src/physics/photon_conservation_geometry.c)
set_property(TARGET test_photon_conservation_index PROPERTY C_STANDARD 99)
target_include_directories(test_photon_conservation_index PRIVATE ${CMAKE_SOURCE_DIR}/src/physics)
target_link_libraries(test_photon_conservation_index PRIVATE m)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_photon_conservation_index PRIVATE -UNDEBUG)
endif()
target_compile_options(test_photon_conservation_index PRIVATE ${SANITIZE_FLAGS})
target_link_options(test_photon_conservation_index PRIVATE ${SANITIZE_FLAGS})
add_test(NAME test_photon_conservation_index COMMAND test_photon_conservation_index)

# MPI changes only read-only spatial queries; compare the full allocation and
# shape certificates with the serial implementation, including failure paths.
add_executable(test_photon_conservation_mpi
    test_photon_conservation_mpi.c
    ${CMAKE_SOURCE_DIR}/src/core/photon_conservation_mpi.c
    ${CMAKE_SOURCE_DIR}/src/physics/photon_conservation_geometry.c
    ${CMAKE_SOURCE_DIR}/src/physics/photon_conservation.c
    ${CMAKE_SOURCE_DIR}/src/physics/photon_conservation_es.c
    ${CMAKE_SOURCE_DIR}/src/physics/photon_conservation_shape.c)
set_property(TARGET test_photon_conservation_mpi PROPERTY C_STANDARD 99)
target_include_directories(test_photon_conservation_mpi PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_photon_conservation_mpi PRIVATE MPI::MPI_C m)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_photon_conservation_mpi PRIVATE -UNDEBUG)
endif()
target_compile_options(test_photon_conservation_mpi PRIVATE ${SANITIZE_FLAGS})
target_link_options(test_photon_conservation_mpi PRIVATE ${SANITIZE_FLAGS})
foreach(pc_ranks 1 2 4)
    add_test(NAME test_photon_conservation_mpi${pc_ranks}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${pc_ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:test_photon_conservation_mpi> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_photon_conservation_mpi${pc_ranks} PROPERTIES
        PROCESSORS ${pc_ranks} TIMEOUT 120 ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1")
endforeach()

add_executable(test_photon_conservation_feedback test_photon_conservation_feedback.c)
set_property(TARGET test_photon_conservation_feedback PROPERTY C_STANDARD 99)
target_include_directories(test_photon_conservation_feedback PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(test_photon_conservation_feedback PRIVATE meraxes_lib)
foreach(pc_ranks 1 2)
    add_test(NAME test_photon_conservation_feedback_mpi${pc_ranks}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${pc_ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:test_photon_conservation_feedback> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_photon_conservation_feedback_mpi${pc_ranks} PROPERTIES PROCESSORS ${pc_ranks})
endforeach()

# Real collective HDF5 reader with a self-contained catalogue fixture.
add_executable(test_velociraptor_reader test_velociraptor_reader.c)
set_property(TARGET test_velociraptor_reader PROPERTY C_STANDARD 99)
target_include_directories(test_velociraptor_reader PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
target_link_libraries(test_velociraptor_reader PRIVATE meraxes_lib)
foreach(reader_ranks 1 2)
    add_test(NAME test_velociraptor_reader_mpi${reader_ranks}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${reader_ranks} ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:test_velociraptor_reader> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(test_velociraptor_reader_mpi${reader_ranks} PROPERTIES
        PROCESSORS ${reader_ranks} TIMEOUT 60)
endforeach()

# Regression tests on the L10_N512 test set (no Criterion needed)
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy, scipy, h5py"
        RESULT_VARIABLE ES_PYTHON_STATUS OUTPUT_QUIET ERROR_QUIET)
    if(ES_PYTHON_STATUS EQUAL 0)
        set(ES_KERNEL_TEST_ENV "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;OMP_WAIT_POLICY=PASSIVE")
        if(TARGET es_hhe_kernels)
            list(APPEND ES_KERNEL_TEST_ENV "ES_NATIVE_LIBRARY=$<TARGET_FILE:es_hhe_kernels>"
                 "ES_NATIVE_REQUIRED=1" "ES_WORKER_THREADS=1")
            foreach(es_kernel_test native_geometry native_chemistry native_backend native_joint native_candidates)
                add_test(NAME test_photon_es_${es_kernel_test}
                    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/es_hhe/test_${es_kernel_test}.py)
                set_tests_properties(test_photon_es_${es_kernel_test} PROPERTIES TIMEOUT 180
                    ENVIRONMENT "${ES_KERNEL_TEST_ENV}")
            endforeach()
        endif()
        add_test(NAME test_photon_es_worker
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/es_hhe/test_snapshot_worker.py)
        set_tests_properties(test_photon_es_worker PROPERTIES TIMEOUT 180
            ENVIRONMENT "${ES_KERNEL_TEST_ENV}")
        foreach(es_python_test photon_floor sparse_geometry)
            add_test(NAME test_photon_es_${es_python_test}
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/es_hhe/test_${es_python_test}.py)
            set_tests_properties(test_photon_es_${es_python_test} PROPERTIES TIMEOUT 180
                ENVIRONMENT "${ES_KERNEL_TEST_ENV}")
        endforeach()
        add_test(NAME test_photon_es_runner
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/es_hhe/test_gadi_runner.py)
        set_tests_properties(test_photon_es_runner PROPERTIES TIMEOUT 120
            ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;ES_PARAMCHECK_BINARY=$<TARGET_FILE:meraxes>")
        add_executable(test_photon_es_adapter test_photon_es_adapter.c)
        set_property(TARGET test_photon_es_adapter PROPERTY C_STANDARD 99)
        target_include_directories(test_photon_es_adapter PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})
        target_link_libraries(test_photon_es_adapter PRIVATE meraxes_lib)
        if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test_photon_es_adapter PRIVATE -UNDEBUG -fno-fast-math -ffp-contract=off)
        endif()
        foreach(es_ranks 1 2 4)
            add_test(NAME test_photon_es_adapter_${es_ranks}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${es_ranks} ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:test_photon_es_adapter> ${MPIEXEC_POSTFLAGS}
                    ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/es_hhe/snapshot_worker.py)
            set_tests_properties(test_photon_es_adapter_${es_ranks} PROPERTIES
                PROCESSORS ${es_ranks} TIMEOUT 180
                ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1;${ES_KERNEL_TEST_ENV}")
        endforeach()
    else()
        message(STATUS "Spectral ES bridge tests disabled: Python needs numpy, scipy and h5py")
    endif()
endif()
add_subdirectory(regression)
