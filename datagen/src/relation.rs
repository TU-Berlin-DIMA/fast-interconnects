// Copyright 2019-2022 Clemens Lutz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Data set generators for generating database relations.
//!
//! The generators produce relation attributes following a random distribution.

use num_traits::FromPrimitive;

use crate::error::{ErrorKind, Result};

use std::convert::TryFrom;
use std::ops::Range;

use rand::distributions::{Distribution, Uniform};
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

use rayon::prelude::*;

use zipf::ZipfDistribution;

/// Specifies that the type is suitable to be a join, grouping, or partitioning key.
///
/// A key attribute is a primitive type (e.g., an integer or floating point type).
/// It reserves a `NULL` value in it's value range and `usize` values map to a
/// unique key value.
///
/// The `NULL` value is expected to have a binary representation of all ones. For
/// signed integers, that value equals -1, for unsigned integers, the value
/// equals 0xF...F.
///
/// The `NULL` value in Rust must be kept in sync with the `NULL` value in C++
/// and CUDA.
pub trait KeyAttribute: Sized + 'static {
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
    /// 0 and ends before, i.e. excluding, attr.len(). Keys are placed at random
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
            .zip(0..)
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
    /// 0 and ends before, i.e. excluding, attr.len(). Keys are placed at random
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
        let mut shuffled: Vec<(usize, T)> = (0..(attr.len()))
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
    pub fn gen_attr<T: FromPrimitive>(attr: &mut [T], range: Range<usize>) -> Result<()> {
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
        range: Range<usize>,
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
    /// The generated values are sampled from 0 to num_elements (exclusive).
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

        // ZipfDistribution generates elements in range [1, num_elements]. Thus,
        // need to substract 1 to get a range [0, num_elements[.
        attr.par_iter_mut()
            .map_init(
                || thread_rng(),
                |mut rng, x| {
                    FromPrimitive::from_usize(between.sample(&mut rng) - 1)
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
