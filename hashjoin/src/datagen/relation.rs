/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright 2019 German Research Center for Artificial Intelligence (DFKI)
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

//! Data set generators for generating database relations.
//!
//! The generators produce relation attributes following a random distribution.

use num_traits::FromPrimitive;

use numa_gpu::error::{ErrorKind, Result};

use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::thread_rng;

use zipf::ZipfDistribution;

pub trait Integer {}

impl Integer for i64 {}

/// Generator for relations with uniform distribution.
pub struct UniformRelation;

impl UniformRelation {
    /// Generates a primary key attribute.
    ///
    /// The generated keys are unique and contiguous. The key range starts from
    /// 1 and ends at, i.e. including, attr.len(). Keys are placed at random
    /// locations within the slice.
    pub fn gen_primary_key<T: FromPrimitive>(attr: &mut [T]) -> Result<()> {
        let mut rng = thread_rng();

        attr.iter_mut()
            .by_ref()
            .zip(1..)
            .map(|(x, i)| {
                FromPrimitive::from_usize(i)
                    .ok_or_else(|| {
                        ErrorKind::IntegerOverflow("Failed to convert from usize".to_string())
                            .into()
                    })
                    .map(|i| *x = i)
            })
            .collect::<Result<()>>()?;

        attr.shuffle(&mut rng);
        Ok(())
    }

    /// Generates a foreign key attribute based on a primary key attribute.
    ///
    /// The generated keys are sampled from the primary key attribute, that is,
    /// they follow a foreign-key relationship. If the primary keys are unique,
    /// then the generated foreign keys follow a uniform distribution.
    pub fn gen_foreign_key_from_primary_key<T: Copy>(fk_attr: &mut [T], pk_attr: &[T]) {
        let mut rng = thread_rng();

        fk_attr
            .iter_mut()
            .by_ref()
            .zip(pk_attr.iter().cycle())
            .for_each(|(fk, pk)| *fk = *pk);
        fk_attr.shuffle(&mut rng);
    }

    /// Generates a uniformly distributed attribute.
    ///
    /// The generated values are sampled from 1 to num_elements (inclusive).
    pub fn gen_attr<T: FromPrimitive + SampleUniform>(
        attr: &mut [T],
        num_elements: usize,
    ) -> Result<()> {
        let mut rng = thread_rng();
        let between = Uniform::from(1..=num_elements);

        attr.iter_mut()
            .by_ref()
            .map(|x| {
                FromPrimitive::from_usize(between.sample(&mut rng))
                    .ok_or_else(|| {
                        ErrorKind::IntegerOverflow("Failed to convert from usize".to_string())
                            .into()
                    })
                    .map(|r| *x = r)
            })
            .collect::<Result<()>>()?;

        Ok(())
    }
}

/// Generator for relations with Zipf distribution.
pub struct ZipfRelation;

impl ZipfRelation {
    /// Generates an attribute following the Zipf distribution.
    ///
    /// The generated values are sampled from 1 to num_elements (inclusive).
    /// Note that the exponent must be greather than 0.
    ///
    /// In the literature, num_elements is also called the alphabet size.
    pub fn gen_attr<T: FromPrimitive>(
        attr: &mut [T],
        num_elements: usize,
        exponent: f64,
    ) -> Result<()> {
        let mut rng = thread_rng();
        let between = ZipfDistribution::new(num_elements, exponent).map_err(|_| {
            ErrorKind::InvalidArgument(
                "ZipfDistribution requires num_elements and exponent greater than 0".to_string(),
            )
        })?;

        attr.iter_mut()
            .by_ref()
            .map(|x| {
                FromPrimitive::from_usize(between.sample(&mut rng))
                    .ok_or_else(|| {
                        ErrorKind::IntegerOverflow("Failed to convert from usize".to_string())
                            .into()
                    })
                    .map(|r| *x = r)
            })
            .collect::<Result<()>>()?;

        Ok(())
    }
}
