/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

//! A collection of data set generators for data sets frequently found in
//! published papers.

use super::relation::{KeyAttribute, UniformRelation};
use crate::error::Result;
use num_traits::FromPrimitive;

/// Generator for the Kim data set.
///
/// The Kim data set is taken from the paper Kim et al. "Sort vs. hash revisited:
/// Fast join implementation on modern multi-core CPUs" in PVLDB 2009.
///
/// The paper uses 4-byte keys / 8-byte tuples.
pub struct Kim;

impl Kim {
    /// Rows in the primary key relation.
    pub fn primary_key_len() -> usize {
        128 * 10_usize.pow(6)
    }

    /// Rows in the foreign key relation.
    pub fn foreign_key_len() -> usize {
        128 * 10_usize.pow(6)
    }

    /// Generate the Kim data set.
    ///
    /// Requires a slice for the primary key attribute, and a slice for the
    /// foreign key attribute. Both slices must have the lengths specified by
    /// the primary_key_len() and foreign_key_len() functions.
    ///
    /// `selectivity` specifies the join selectivity in percent. An according
    /// percentage of keys are set to the `NULL` value. By default (`None`), the
    /// selectivity is 100%.
    pub fn gen<T: Copy + Send + KeyAttribute + FromPrimitive>(
        pk_attr: &mut [T],
        fk_attr: &mut [T],
        selectivity: Option<u32>,
    ) -> Result<()> {
        assert!(pk_attr.len() == Self::primary_key_len());
        assert!(fk_attr.len() == Self::foreign_key_len());

        UniformRelation::gen_primary_key_par(pk_attr, selectivity)?;
        UniformRelation::gen_attr_par(fk_attr, 0..pk_attr.len())?;
        Ok(())
    }
}

/// Generator for the Blanas data set.
///
/// The Blanas data set is taken from the paper Blanas et al. "Design and
/// evaluation of main memory hash join algorithms for multi-core CPUs" in
/// SIGMOD 2011.
///
/// The paper uses 8-byte keys / 16-byte tuples.
pub struct Blanas;

impl Blanas {
    /// Rows in the primary key relation.
    pub fn primary_key_len() -> usize {
        16 * 2_usize.pow(20)
    }

    /// Rows in the foreign key relation.
    pub fn foreign_key_len() -> usize {
        256 * 2_usize.pow(20)
    }

    /// Generate the Blanas data set.
    ///
    /// Requires a slice for the primary key attribute, and a slice for the
    /// foreign key attribute. Both slices must have the lengths specified by
    /// the primary_key_len() and foreign_key_len() functions.
    ///
    /// `selectivity` specifies the join selectivity in percent. An according
    /// percentage of keys are set to the `NULL` value. By default (`None`), the
    /// selectivity is 100%.
    pub fn gen<T: Copy + Send + KeyAttribute + FromPrimitive>(
        pk_attr: &mut [T],
        fk_attr: &mut [T],
        selectivity: Option<u32>,
    ) -> Result<()> {
        assert!(pk_attr.len() == Self::primary_key_len());
        assert!(fk_attr.len() == Self::foreign_key_len());

        UniformRelation::gen_primary_key_par(pk_attr, selectivity)?;
        UniformRelation::gen_attr_par(fk_attr, 0..pk_attr.len())?;
        Ok(())
    }
}
