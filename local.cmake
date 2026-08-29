set(FFTW_LIBRARY "/apps/fftw3/3.3.8/lib/GNU/libfftw3f.so" CACHE FILEPATH "" FORCE)
set(FFTW_MPI_LIBRARY "/apps/fftw3/3.3.8/lib/ompi3/GNU/libfftw3f_mpi.so" CACHE FILEPATH "" FORCE)

# The hdf5/1.10.5p module only ships MPI-flavoured library names
# (libhdf5_ompi3.so etc.), not a plain libhdf5.so, so CMake's FindHDF5
# resolves the compiler wrapper (h5pcc) but still emits a bare "-lhdf5"
# at link time that the linker can't find. Point straight at the real
# files (per `h5pcc -show`).
set(HDF5_C_LIBRARY_hdf5 "/apps/hdf5/1.10.5p/lib/ompi3/libhdf5.so" CACHE FILEPATH "" FORCE)
set(HDF5_C_LIBRARY_hdf5_hl "/apps/hdf5/1.10.5p/lib/ompi3/libhdf5_hl.so" CACHE FILEPATH "" FORCE)
