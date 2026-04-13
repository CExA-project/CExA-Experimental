FROM mcr.microsoft.com/vscode/devcontainers/cpp:ubuntu-22.04

# Install C++ development tools, Git and CMake 
RUN apt-get update
RUN apt-get -y install \
    build-essential \
    gdb \
    git \
    cmake

RUN git clone https://github.com/kokkos/kokkos.git \
    --branch release-candidate-4.6.01 --depth=1 /opt/kokkos && \
    cd /opt/kokkos && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/opt/kokkos-install \
             -DKokkos_ENABLE_OPENMP=ON \
             -DKokkos_ENABLE_SERIAL=ON \
             -DCMAKE_CXX_STANDARD=17 && \
    cmake --build . --parallel $(( $(nproc) / 2))  && \
    cmake --build . --target install

ENV CPATH="/opt/kokkos-install/include:$CPATH"
