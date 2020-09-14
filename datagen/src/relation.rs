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

use crate::error::{ErrorKind, Result};

use std::convert::TryFrom;
use std::ops::RangeInclusive;

use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

use rayon::prelude::*;

use zipf::ZipfDistribution;

pub trait KeyAttribute: Sized {
    fn null_key() -> Self;
    fn try_from_usize(x: usize) -> Result<Self>;
}

impl KeyAttribute for i32 {
    fn null_key() -> Self {
        -1
    }

    fn try_from_usize(x: usize) -> Result<Self> {
        Self::try_from(x).map_err(|_| {
            ErrorKind::IntegerOverflow("Failed to covnert from usize".to_string()).into()
        })
    }
}

impl KeyAttribute for i64 {
    fn null_key() -> Self {
        -1
    }

    fn try_from_usize(x: usize) -> Result<Self> {
        Self::try_from(x).map_err(|_| {
            ErrorKind::IntegerOverflow("Failed to covnert from usize".to_string()).into()
        })
    }
}

/// Generator for relations with uniform distribution.
pub struct UniformRelation;

impl UniformRelation {
    /// Generates a primary key attribute.
    ///
    /// The generated keys are unique and contiguous. The key range starts from
    /// 1 and ends at, i.e. including, attr.len(). Keys are placed at random
    /// locations within the slice.
    ///
    /// `selectivity` specifies the join selectivity in percent. An according
    /// percentage of keys are set to the `NULL` value. By default (`None`), the
    /// selectivity is 100%.
    pub fn gen_primary_key<T: KeyAttribute>(
        attr: &mut [T],
        selectivity: Option<u32>,
    ) -> Result<()> {
        let selectivity = selectivity.unwrap_or_else(|| 100);
        let percent = Uniform::from(0..=100);
        let mut rng = thread_rng();

        attr.iter_mut()
            .by_ref()
            .zip(1..)
            .map(|(x, i)| {
                T::try_from_usize(i).map(|i| {
                    let val = if percent.sample(&mut rng) <= selectivity {
                        i
                    } else {
                        T::null_key()
                    };
                    *x = val
                })
            })
            .collect::<Result<()>>()?;

        attr.shuffle(&mut rng);
        Ok(())
    }

    /// Generates a primary key attribute in parallel.
    ///
    /// The generated keys are unique and contiguous. The key range starts from
    /// 1 and ends at, i.e. including, attr.len(). Keys are placed at random
    /// locations within the slice.
    ///
    /// `selectivity` specifies the join selectivity in percent. An according
    /// percentage of keys are set to the `NULL` value. By default (`None`), the
    /// selectivity is 100%.
    pub fn gen_primary_key_par<T: Clone + Send + KeyAttribute>(
        attr: &mut [T],
        selectivity: Option<u32>,
    ) -> Result<()> {
        let selectivity = selectivity.unwrap_or_else(|| 100);
        let percent = Uniform::from(0..=100);
        let mut shuffled: Vec<(usize, T)> = (1..(attr.len() + 1))
            .into_par_iter()
            .map_init(thread_rng, |mut rng, i| {
                T::try_from_usize(i).map(|i| {
                    let val = if percent.sample(&mut rng) <= selectivity {
                        i
                    } else {
                        T::null_key()
                    };
                    (rng.gen(), val)
                })
            })
            .collect::<Result<_>>()?;

        shuffled.as_mut_slice().par_sort_unstable_by_key(|x| x.0);

        attr.par_iter_mut()
            .zip_eq(shuffled.into_par_iter())
            .for_each(|(x, t)| *x = t.1);

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
    /// The generated values are sampled from `range`.
    pub fn gen_attr<T: FromPrimitive>(attr: &mut [T], range: RangeInclusive<usize>) -> Result<()> {
        let mut rng = thread_rng();
        let between = Uniform::from(range);

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

    /// Generates a uniformly distributed attribute in parallel.
    ///
    /// The generated values are sampled from `range`.
    pub fn gen_attr_par<T: FromPrimitive + Send>(
        attr: &mut [T],
        range: RangeInclusive<usize>,
    ) -> Result<()> {
        let between = Uniform::from(range);

        attr.par_iter_mut()
            .map_init(
                || thread_rng(),
                |mut rng, x| {
                    FromPrimitive::from_usize(between.sample(&mut rng))
                        .ok_or_else(|| {
                            ErrorKind::IntegerOverflow("Failed to convert from usize".to_string())
                                .into()
                        })
                        .map(|r| *x = r)
                },
            )
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

    /// Generates an attribute following the Zipf distribution in parallel.
    ///
    /// The generated values are sampled from 1 to num_elements (inclusive).
    /// Note that the exponent must be greather than 0.
    ///
    /// In the literature, num_elements is also called the alphabet size.
    pub fn gen_attr_par<T: FromPrimitive + Send>(
        attr: &mut [T],
        num_elements: usize,
        exponent: f64,
    ) -> Result<()> {
        let between = ZipfDistribution::new(num_elements, exponent).map_err(|_| {
            ErrorKind::InvalidArgument(
                "ZipfDistribution requires num_elements and exponent greater than 0".to_string(),
            )
        })?;

        attr.par_iter_mut()
            .map_init(
                || thread_rng(),
                |mut rng, x| {
                    FromPrimitive::from_usize(between.sample(&mut rng))
                        .ok_or_else(|| {
                            ErrorKind::IntegerOverflow("Failed to convert from usize".to_string())
                                .into()
                        })
                        .map(|r| *x = r)
                },
            )
            .collect::<Result<()>>()?;

        Ok(())
    }
}
