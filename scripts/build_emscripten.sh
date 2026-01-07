#!/bin/bash

set -e
rm -rf cm* CM* lib* unisamp* Testing* tests* include tests utils Make*
emcmake cmake -DCMAKE_INSTALL_PREFIX=$EMINSTALL -DENABLE_TESTING=OFF ..
emmake make -j26
emmake make install
cp unisamp.wasm ../html
cp $EMINSTALL/bin/unisamp.js ../html
