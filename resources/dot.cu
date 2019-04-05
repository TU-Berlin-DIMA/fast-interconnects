/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

extern "C"
__global__
void dot(size_t const len, float const *x, float const *y, float *r) {
    float tmp = 0.0;
    for (
            size_t i = blockIdx.x * blockDim.x + threadIdx.x;
            i < len;
            i += blockDim.x * gridDim.x
        )
    {
        tmp += x[i] * y[i];
    }

    atomicAdd(r, tmp);
}
