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
add_subdirectory(regression)
