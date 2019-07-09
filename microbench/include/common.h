/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2018, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef COMMON_H
#define COMMON_H

#include <cstdint>

enum PingPong : uint32_t
{
    NONE = 0,
    CPU,
    GPU
};

enum Signal : uint32_t
{
    WAIT = 0,
    START
};

// POWER9 has 128-byte cache lines
// That equals 32 4-byte ints
struct CacheLine {
    PingPong value;
    uint32_t other[31];
};

#endif /* COMMON_H */
