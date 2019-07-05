/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2018 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

extern crate crossbeam_utils;
#[macro_use]
extern crate error_chain;
#[macro_use]
extern crate rustacuda;
extern crate rayon;

pub mod datagen;
pub mod error;
pub mod operators;
pub mod runtime;
