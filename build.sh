cd build
echo Starting build at $(date)
echo ------------------------------------------
cmake -DBUILD_PERF_TESTS:BOOL=OFF -DBUILD_opencv_world=OFF -DBUILD_opencv_python3=ON -DBUILD_TESTS:BOOL=OFF -DBUILD_DOCS:BOOL=OFF  -DWITH_CUDA:BOOL=OFF -DBUILD_EXAMPLES:BOOL=OFF -DINSTALL_CREATE_DISTRIB=ON -DCMAKE_BUILD_TYPE=Debug ../opencv/
make -j8
sudo make install