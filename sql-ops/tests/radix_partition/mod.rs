/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2020-2021 Clemens Lutz, German Research Center for Artificial
 * Intelligence
 * Author: Clemens Lutz, DFKI GmbH <clemens.lutz@dfki.de>
 */

use rustacuda::memory::DeviceCopy;
use sql_ops::partition::{PartitionedRelation, RadixBits, RadixPass, Tuple};
use std::collections::hash_map::{Entry, HashMap};
use std::error::Error;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::iter;
use std::result::Result;

pub fn key_to_partition(key: i32, radix_bits: &RadixBits, radix_pass: RadixPass) -> u32 {
    let ignore_bits = radix_bits.pass_ignore_bits(radix_pass);
    let fanout = radix_bits.pass_fanout(radix_pass).unwrap();
    let mask = (fanout - 1) << ignore_bits;
    let partition = (key as u32 & mask) >> ignore_bits;
    partition
}

pub fn tuple_loss_or_duplicates<T>(
    _radix_pass: RadixPass,
    _radix_bits: &RadixBits,
    data_key: &[T],
    data_pay: &[T],
    partitioned_relation: &PartitionedRelation<Tuple<T, T>>,
    partition_id: Option<u32>,
) -> Result<(), Box<dyn Error>>
where
    T: Clone + Debug + Default + Display + DeviceCopy + Eq + Hash,
{
    let mut original_tuples: HashMap<_, _> = data_key
        .iter()
        .cloned()
        .zip(data_pay.iter().cloned().zip(std::iter::repeat(0)))
        .collect();

    let relation = unsafe { partitioned_relation.as_raw_relation_slice()? };

    relation.iter().cloned().for_each(|Tuple { key, value }| {
        let entry = original_tuples.entry(key);
        match entry {
            entry @ Entry::Occupied(_) => {
                let key = entry.key().clone();
                entry.and_modify(|(original_value, counter)| {
                    let id_str = partition_id
                        .map_or_else(|| "".to_string(), |id| format!(" in partition {}", id));
                    assert_eq!(
                        value, *original_value,
                        "Invalid payload{}: {}; expected: {}",
                        id_str, value, *original_value
                    );
                    assert_eq!(*counter, 0, "Duplicate key: {}", key);
                    *counter = *counter + 1;
                });
            }
            entry @ Entry::Vacant(_) => {
                // skip padding entries
                if *entry.key() != T::default() {
                    assert!(false, "Invalid key: {}", entry.key());
                }
            }
        };
    });

    original_tuples.iter().for_each(|(key, &(_, counter))| {
        assert_eq!(
            counter, 1,
            "Key {} occurs {} times; expected exactly once",
            key, counter
        );
    });

    Ok(())
}

// FIXME: Convert i32 to a generic type
pub fn verify_partitions(
    radix_pass: RadixPass,
    radix_bits: &RadixBits,
    _data_key: &[i32],
    _data_pay: &[i32],
    partitioned_relation: &PartitionedRelation<Tuple<i32, i32>>,
    partition_id: Option<u32>,
) -> Result<(), Box<dyn Error>> {
    (0..partitioned_relation.num_chunks())
        .flat_map(|c| iter::repeat(c).zip(0..partitioned_relation.fanout()))
        .flat_map(|(c, p)| iter::repeat((c, p)).zip(partitioned_relation[(c, p)].iter()))
        .enumerate()
        .for_each(|(i, ((c, p), &tuple))| {
            let dst_partition = key_to_partition(tuple.key, radix_bits, radix_pass);
            let id_str = partition_id.map_or_else(|| "".to_string(), |id| format!("{}:", id));
            assert_eq!(
                dst_partition, p,
                "Wrong partitioning detected in chunk {} at position {}: \
                key {} in partition {}{}; expected partition {}{}",
                c, i, tuple.key, id_str, p, id_str, dst_partition
            );
        });

    Ok(())
}

pub fn check_copy_with_payload(
    _radix_pass: RadixPass,
    _radix_bits: &RadixBits,
    data_key: &[i32],
    data_pay: &[i32],
    _partitioned_relation: &PartitionedRelation<Tuple<i32, i32>>,
    cached_key: &[i32],
    cached_pay: &[i32],
) -> Result<(), Box<dyn Error>> {
    data_key
        .iter()
        .zip(cached_key.iter())
        .enumerate()
        .for_each(|(i, (&original, &cached))| {
            assert_eq!(
                original, cached,
                "Wrong key detected at position {}: {}",
                i, cached
            );
        });

    data_pay
        .iter()
        .zip(cached_pay.iter())
        .enumerate()
        .for_each(|(i, (&original, &cached))| {
            assert_eq!(
                original, cached,
                "Wrong payload detected at position {}: {}",
                i, cached
            );
        });

    Ok(())
}

pub fn two_pass_tuple_loss_or_duplicates(
    radix_pass: RadixPass,
    radix_bits: &RadixBits,
    _partitioned_relation_1st: &PartitionedRelation<Tuple<i32, i32>>,
    partition_id: u32,
    partitioned_relation_2nd: &PartitionedRelation<Tuple<i32, i32>>,
    cached_key_slice: &[i32],
    cached_pay_slice: &[i32],
) -> Result<(), Box<dyn Error>> {
    tuple_loss_or_duplicates(
        radix_pass,
        radix_bits,
        cached_key_slice,
        cached_pay_slice,
        partitioned_relation_2nd,
        Some(partition_id),
    )
}

pub fn two_pass_verify_partitions(
    radix_pass: RadixPass,
    radix_bits: &RadixBits,
    _partitioned_relation_1st: &PartitionedRelation<Tuple<i32, i32>>,
    partition_id: u32,
    partitioned_relation_2nd: &PartitionedRelation<Tuple<i32, i32>>,
    cached_key_slice: &[i32],
    cached_pay_slice: &[i32],
) -> Result<(), Box<dyn Error>> {
    verify_partitions(
        radix_pass,
        radix_bits,
        cached_key_slice,
        cached_pay_slice,
        partitioned_relation_2nd,
        Some(partition_id),
    )
}

pub fn verify_transformed_input(
    _radix_pass: RadixPass,
    _radix_bits: &RadixBits,
    partitioned_relation_1st: &PartitionedRelation<Tuple<i32, i32>>,
    partition_id: u32,
    _partitioned_relation_2nd: &PartitionedRelation<Tuple<i32, i32>>,
    cached_key_slice: &[i32],
    cached_pay_slice: &[i32],
) -> Result<(), Box<dyn Error>> {
    (0..partitioned_relation_1st.num_chunks())
        .flat_map(|c| partitioned_relation_1st[(c, partition_id)].iter())
        .zip(cached_key_slice.iter())
        .zip(cached_pay_slice.iter())
        .enumerate()
        .for_each(|(i, ((&tuple, &key), &pay))| {
            assert_eq!(
                tuple.key, key,
                "Wrong key detected in partition {} at position {}: {}",
                partition_id, i, key
            );
            assert_eq!(
                tuple.value, pay,
                "Wrong payload detected in partition {} at position {}: {}",
                partition_id, i, pay
            );
        });

    Ok(())
}
