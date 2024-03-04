#!/bin/bash

# This script will download external module files from fiesta, magpie, mpiutils and shift
# to be used internally in cactus and avoid unnecessary dependencies.

echo " "
echo " Synchronising external modules"
echo " "

echo " Downloading FIESTA files"
echo " "

cd cactus/ext/fiesta

fiestalist="mpi_periodic mpi_randoms periodic randoms"

cd boundary

for fiestafile in $fiestalist ;
  do
    echo " "
    echo " Downloading ${fiestafile}.py from https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/boundary/${fiestafile}.py"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/boundary/${fiestafile}.py
    if test -f ${fiestafile}.py.1; then
      echo " "
      echo " ext/fiesta/boundary/${fiestafile}.py was downloaded"
      echo " "
      rm ${fiestafile}.py
      mv ${fiestafile}.py.1 ${fiestafile}.py
    else
      echo " "
      echo " ERROR: ext/fiesta/boundary/${fiestafile}.py was not downloaded!"
      echo " "
    fi
  done

cd ..

fiestalist="mpi_points points"

cd coords

for fiestafile in $fiestalist ;
  do
    echo " "
    echo " Downloading ${fiestafile}.py from https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/coords/${fiestafile}.py"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/coords/${fiestafile}.py
    if test -f ${fiestafile}.py.1; then
      echo " "
      echo " ext/fiesta/coords/${fiestafile}.py was downloaded"
      echo " "
      rm ${fiestafile}.py
      mv ${fiestafile}.py.1 ${fiestafile}.py
    else
      echo " "
      echo " ERROR: ext/fiesta/coords/${fiestafile}.py was not downloaded!"
      echo " "
    fi
  done

cd ..

fiestalist="dtfe2d dtfe3d dtfe4grid mpi_dtfe4grid"

cd dtfe

for fiestafile in $fiestalist ;
  do
    echo " "
    echo " Downloading ${fiestafile}.py from https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/dtfe/${fiestafile}.py"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/dtfe/${fiestafile}.py
    if test -f ${fiestafile}.py.1; then
      echo " "
      echo " ext/fiesta/dtfe/${fiestafile}.py was downloaded"
      echo " "
      rm ${fiestafile}.py
      mv ${fiestafile}.py.1 ${fiestafile}.py
    else
      echo " "
      echo " ERROR: ext/fiesta/dtfe/${fiestafile}.py was not downloaded!"
      echo " "
    fi
  done

cd ..

fiestalist="diff mpi_diff"

cd maths

for fiestafile in $fiestalist ;
  do
    echo " "
    echo " Downloading ${fiestafile}.py from https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/maths/${fiestafile}.py"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/maths/${fiestafile}.py
    if test -f ${fiestafile}.py.1; then
      echo " "
      echo " ext/fiesta/maths/${fiestafile}.py was downloaded"
      echo " "
      rm ${fiestafile}.py
      mv ${fiestafile}.py.1 ${fiestafile}.py
    else
      echo " "
      echo " ERROR: ext/fiesta/maths/${fiestafile}.py was not downloaded!"
      echo " "
    fi
  done

cd ..

fiestalist="deconvol mpi_part2grid part2grid"

cd p2g

for fiestafile in $fiestalist ;
  do
    echo " "
    echo " Downloading ${fiestafile}.py from https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/p2g/${fiestafile}.py"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/p2g/${fiestafile}.py
    if test -f ${fiestafile}.py.1; then
      echo " "
      echo " ext/fiesta/p2g/${fiestafile}.py was downloaded"
      echo " "
      rm ${fiestafile}.py
      mv ${fiestafile}.py.1 ${fiestafile}.py
    else
      echo " "
      echo " ERROR: ext/fiesta/p2g/${fiestafile}.py was not downloaded!"
      echo " "
    fi
  done

cd ..

fiestalist="cartesian"

cd randoms

for fiestafile in $fiestalist ;
  do
    echo " "
    echo " Downloading ${fiestafile}.py from https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/randoms/${fiestafile}.py"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/randoms/${fiestafile}.py
    if test -f ${fiestafile}.py.1; then
      echo " "
      echo " ext/fiesta/randoms/${fiestafile}.py was downloaded"
      echo " "
      rm ${fiestafile}.py
      mv ${fiestafile}.py.1 ${fiestafile}.py
    else
      echo " "
      echo " ERROR: ext/fiesta/randoms/${fiestafile}.py was not downloaded!"
      echo " "
    fi
  done

cd ..

fiestalist="delaunay2d delaunay3d differentiate grid matrix part2grid_2d part2grid_3d
part2grid_pix part2grid_wei polygon polyhedron"

cd src

for fiestafile in $fiestalist ;
  do
    echo " "
    echo " Downloading ${fiestafile}.py from https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/src/${fiestafile}.f90"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/src/${fiestafile}.f90
    if test -f ${fiestafile}.f90.1; then
      echo " "
      echo " ext/fiesta/src/${fiestafile}.f90 was downloaded"
      echo " "
      rm ${fiestafile}.f90
      mv ${fiestafile}.f90.1 ${fiestafile}.f90
    else
      echo " "
      echo " ERROR: ext/fiesta/src/${fiestafile}.f90 was not downloaded!"
      echo " "
    fi
  done

cd ..

fiestalist="complex lists vectors"

cd utils

for fiestafile in $fiestalist ;
  do
    echo " "
    echo " Downloading ${fiestafile}.py from https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/utils/${fiestafile}.py"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/fiesta/development/fiesta/utils/${fiestafile}.py
    if test -f ${fiestafile}.py.1; then
      echo " "
      echo " ext/fiesta/utils/${fiestafile}.py was downloaded"
      echo " "
      rm ${fiestafile}.py
      mv ${fiestafile}.py.1 ${fiestafile}.py
    else
      echo " "
      echo " ERROR: ext/fiesta/utils/${fiestafile}.py was not downloaded!"
      echo " "
    fi
  done

cd ../../../..

echo " Downloading MAGPIE files"
echo " "

cd cactus/ext/magpie

magpielist="index_unique pix2pos_cart pos2pix_cart"

cd pixels

for magpiefile in $magpielist ;
  do
    echo " "
    echo " Downloading ${magpiefile}.py from https://raw.githubusercontent.com/knaidoo29/magpie/development/magpie/pixels/${magpiefile}.py"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/magpie/development/magpie/pixels/${magpiefile}.py
    if test -f ${magpiefile}.py.1; then
      echo " "
      echo " ext/magpie/pixels/${magpiefile}.py was downloaded"
      echo " "
      rm ${magpiefile}.py
      mv ${magpiefile}.py.1 ${magpiefile}.py
    else
      echo " "
      echo " ERROR: ext/magpie/pixels/${magpiefile}.py was not downloaded!"
      echo " "
    fi
  done

cd ..

magpielist="check progress type"

cd utils

for magpiefile in $magpielist ;
  do
    echo " "
    echo " Downloading ${magpiefile}.py from https://raw.githubusercontent.com/knaidoo29/magpie/development/magpie/utils/${magpiefile}.py"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/magpie/development/magpie/utils/${magpiefile}.py
    if test -f ${magpiefile}.py.1; then
      echo " "
      echo " ext/magpie/utils/${magpiefile}.py was downloaded"
      echo " "
      rm ${magpiefile}.py
      mv ${magpiefile}.py.1 ${magpiefile}.py
    else
      echo " "
      echo " ERROR: ext/magpie/utils/${magpiefile}.py was not downloaded!"
      echo " "
    fi
  done

cd ..

magpielist="pixel_1dto2d pixel_1dto3d pixel_binbyindex pixel_utils"

cd src

for magpiefile in $magpielist ;
  do
    echo " "
    echo " Downloading ${magpiefile}.py from https://raw.githubusercontent.com/knaidoo29/magpie/development/magpie/src/${magpiefile}.f90"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/magpie/development/magpie/src/${magpiefile}.f90
    if test -f ${magpiefile}.f90.1; then
      echo " "
      echo " ext/magpie/src/${magpiefile}.f90 was downloaded"
      echo " "
      rm ${magpiefile}.f90
      mv ${magpiefile}.f90.1 ${magpiefile}.f90
    else
      echo " "
      echo " ERROR: ext/magpie/src/${magpiefile}.f90 was not downloaded!"
      echo " "
    fi
  done

cd ../../../..


echo " Downloading MPIutils files"
echo " "

cd cactus/ext/mpiutils

mpiutilslist="loops mpiclass"

for mpiutilsfile in $mpiutilslist ;
  do
    echo " "
    echo " Downloading ${mpiutilsfile}.py from https://raw.githubusercontent.com/knaidoo29/MPIutils/master/mpiutils/${mpiutilsfile}.py"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/MPIutils/master/mpiutils/${mpiutilsfile}.py
    if test -f ${mpiutilsfile}.py.1; then
      echo " "
      echo " ext/mpiutils/${mpiutilsfile}.py was downloaded"
      echo " "
      rm ${mpiutilsfile}.py
      mv ${mpiutilsfile}.py.1 ${mpiutilsfile}.py
    else
      echo " "
      echo " ERROR: ext/mpiutils/${mpiutilsfile}.py was not downloaded!"
      echo " "
    fi
  done

cd ../../..


echo " Downloading SHIFT files"
echo " "

cd cactus/ext/shift/cart

shiftlist="conv diff fft grid kgrid mpi_fft mpi_kgrid utils"

for shiftfile in $shiftlist ;
  do
    echo " "
    echo " Downloading ${shiftfile}.py from https://raw.githubusercontent.com/knaidoo29/SHIFT/master/shift/cart/${shiftfile}.py"
    echo " "
    wget https://raw.githubusercontent.com/knaidoo29/SHIFT/master/shift/cart/${shiftfile}.py
    if test -f ${shiftfile}.py.1; then
      echo " "
      echo " ext/shift/cart/${shiftfile}.py was downloaded"
      echo " "
      rm ${shiftfile}.py
      mv ${shiftfile}.py.1 ${shiftfile}.py
    else
      echo " "
      echo " ERROR: ext/shift/cart/${shiftfile}.py was not downloaded!"
      echo " "
    fi
  done

cd ../../../..

editfiles="fiesta/coords/mpi_points fiesta/dtfe/dtfe4grid fiesta/dtfe/mpi_dtfe4grid fiesta/p2g/deconvol
fiesta/p2g/mpi_part2grid"

echo " "
echo " Editing files with 'import shift'"
echo " "

for editfile in $editfiles ;
  do
    echo " Editing 'import shift' -> 'from ... import shift' in file cactus/ext/${editfile}.py"
    sed -i '' 's/import shift/from ... import shift/' cactus/ext/${editfile}.py
  done

echo " "
