// Copyright 2020-2022 Clemens Lutz
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

use crate::error::Result;
use datagen::relation::UniformRelation;
use numa_gpu::runtime::allocator::{Allocator, DerefMemType};
use numa_gpu::runtime::memory::DerefMem;
// use rand::distributions::weighted::alias_method::WeightedIndex;
use rand::distributions::{Distribution, Uniform};
use rand_distr::Normal;
use rayon::prelude::*;

/// A materialized LineItem tuple
///
/// Mainly used to calculate the size of a tuple for TPC-H Query 6.
#[repr(packed)]
pub struct LineItemTuple {
    pub shipdate: i32,
    pub discount: i32,
    pub quantity: i32,
    pub extendedprice: i32,
}

/// A columnar LineItem table for Query 6.
///
/// The table contains only those attributes used in Query 6.
///
/// The number of rows in LineItem is specified based on the Orders table.
/// For each row in orders, there are between [1, 7] rows in LineItem. To avoid
/// generating a Orders table, the length of LineItem is calculated based on the
/// Central Limit Theorem. The CLM says that the sum of N variables has a normal
/// distribution with mean = 4 * N and variance = 3, because a uniform
/// distribution between [1, 7] has mean = 4 and variance = 3. N is specified as
/// scale_factor * 1'500'000.
pub struct LineItem {
    /// l_shipdate
    ///
    /// The date is encoded as the number of days starting from 1992-01-01.
    ///
    /// The shipdate is based on o_orderdate. o_orderdate is sampled from dates
    /// between 1992-01-01 and 1998-12-31, minus 151 days. This range includes
    /// 2557-151 = 2406 days, whereby 1992 and 1996 are leap years. The shipdate
    /// then includes a random number of days between [0, 121] added onto the
    /// orderdate.
    ///
    /// Overall, to sample a shipdate, the orderdate is uniformly sampled from
    /// [0, 2406] and the a uniformly sampled value between [0, 121] is added on
    /// top.
    pub shipdate: DerefMem<i32>,

    /// l_discount
    ///
    /// The discount is specified as a fixed-point decimal value between
    /// [0.00, 0.10]. It is encoded as percent, i.e., times 100.
    pub discount: DerefMem<i32>,

    /// l_quantity
    ///
    /// The quantity is specified as a random integer value between [1, 50].
    pub quantity: DerefMem<i32>,

    /// l_extendedprice
    ///
    /// The exact value of extendedprice has no impact on measuring Query 6.
    /// Thus, it is sampled as a random integer between [1, MAX_INT].
    pub extendedprice: DerefMem<i32>,
}

impl LineItem {
    pub fn new(scale_factor: u32, mem_type: DerefMemType) -> Result<LineItem> {
        let alloc = Allocator::deref_mem_alloc_fn::<i32>(mem_type.clone());
        let len_orders: usize = scale_factor as usize * 1_500_000;

        // Calculate lineitem length based on central limit theorem
        let len_lineitem_mean: f64 = len_orders as f64 * 4.0;
        let len_lineitem_variance: f64 = 3.0;
        let normal_distribution = Normal::new(len_lineitem_mean, len_lineitem_variance.sqrt())
            .expect("Failed because standard deviation is too small; Looks like a bug!");
        let mut rng = rand::thread_rng();
        let len_lineitem: usize = normal_distribution.sample(&mut rng) as usize;

        // Generate l_shipdate
        //
        // l_shipdate depends on o_orderdate. o_orderdate is sampled from dates
        // between [0,2557-151] with a uniform random distribution (and another
        // random number is added on top, as per the TPC-H spec). Therefore,
        // l_shipdate can be sampled from the same dates, but the distribution
        // must be weighted with weights between [1,7].
        //
        // Finally, deallocate o_orderdate to free space before allocating the
        // other lineitem attributes.
        // let mut shipdate = alloc(len_lineitem);
        // {
        //     let order_dates = 0..=(2557 - 151);
        //     // let weight_distribution = Uniform::from(1..=7);
        //     // let date_weights: Vec<u8> = order_dates
        //     //     .iter()
        //     //     .map(|_| weight_distribution.sample(&mut rng))
        //     //     .collect();
        //
        //     // let weighted_order_dates: Vec<i32> = order_dates
        //     //     .flat_map(|date| {
        //     //         let weight = weight_distribution.sample(&mut rng);
        //     //         std::iter::repeat(date).take(weight as usize)
        //     //     })
        //     //     .collect();
        //
        //     // let lineitem_distribution = WeightedIndex::new(date_weights)
        //     //     .expect("Failed to generate weighted index. Looks like a bug!");
        //     let shipdate_distribution = Uniform::from(1..=121);
        //     let order_date_distribution = Uniform::from(0..(weighted_order_dates.len()));
        //
        //     shipdate.par_iter_mut().for_each_init(
        //         || rand::thread_rng(),
        //         |mut rng, date| {
        //             *date = weighted_order_dates[order_date_distribution.sample(&mut rng)]
        //                 + shipdate_distribution.sample(&mut rng)
        //         },
        //     );
        // }

        // Generate the other attributes
        let mut shipdate = alloc(len_lineitem);
        let mut discount = alloc(len_lineitem);
        let mut quantity = alloc(len_lineitem);
        let mut extendedprice = alloc(len_lineitem);

        let order_date_distribution = Uniform::from(0..=(2557 - 151));
        let shipdate_distribution = Uniform::from(1..=121);
        shipdate.par_iter_mut().for_each_init(
            || rand::thread_rng(),
            |mut rng, date| {
                *date = order_date_distribution.sample(&mut rng)
                    + shipdate_distribution.sample(&mut rng)
            },
        );

        UniformRelation::gen_attr_par(&mut discount, 0..11)?;
        UniformRelation::gen_attr_par(&mut quantity, 1..51)?;
        UniformRelation::gen_attr_par(&mut extendedprice, 1..(i32::max_value() as usize + 1))?;

        // Finalize lineitem relation
        Ok(Self {
            shipdate,
            discount,
            quantity,
            extendedprice,
        })
    }

    pub fn len(&self) -> usize {
        self.shipdate.len()
    }
}
