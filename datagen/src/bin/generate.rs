/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use datagen::popular;
use serde::ser::Serialize;
use serde_derive::Serialize;
use std::io::Write;
use std::path::PathBuf;
use structopt::clap::arg_enum;
use structopt::StructOpt;

type Result<T> = std::result::Result<T, Box<std::error::Error>>;

fn main() -> Result<()> {
    // Parse commandline arguments
    let cmd = CmdOpt::from_args();

    // Create file
    // FIXME: abstract the serializer type
    let (mut inner_writer, mut outer_writer): (Box<csv::Writer<_>>, Box<csv::Writer<_>>) =
        match cmd.file_type {
            ArgFileType::Csv => {
                let mut spec = csv::WriterBuilder::new();
                spec.has_headers(true).delimiter(b',');
                (
                    Box::new(spec.from_path(&cmd.inner_rel_path)?),
                    Box::new(spec.from_path(&cmd.outer_rel_path)?),
                )
            }
            ArgFileType::Tsv => {
                let mut spec = csv::WriterBuilder::new();
                spec.has_headers(true).delimiter(b' ');
                (
                    Box::new(spec.from_path(&cmd.inner_rel_path)?),
                    Box::new(spec.from_path(&cmd.outer_rel_path)?),
                )
            }
        };

    // Generate
    if let (Some(inner), Some(outer)) = (cmd.inner_rel_tuples, cmd.outer_rel_tuples) {
        match cmd.tuple_bytes {
            ArgTupleBytes::Bytes8 => {
                generate::<i32, _>(inner, outer, &mut inner_writer, &mut outer_writer)?;
            }
            ArgTupleBytes::Bytes16 => {
                generate::<i64, _>(inner, outer, &mut inner_writer, &mut outer_writer)?;
            }
        }
    } else if let Some(data_set) = cmd.data_set {
        match cmd.tuple_bytes {
            ArgTupleBytes::Bytes8 => {
                generate_popular::<i32, _>(data_set, &mut inner_writer, &mut outer_writer)?;
            }
            ArgTupleBytes::Bytes16 => {
                generate_popular::<i64, _>(data_set, &mut inner_writer, &mut outer_writer)?;
            }
        }
    } else {
        unreachable!();
    }

    Ok(())
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq)]
    pub enum ArgDataSet {
        Blanas,
        Kim,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq)]
    #[repr(usize)]
    pub enum ArgTupleBytes {
        Bytes8 = 8,
        Bytes16 = 16,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq)]
    #[repr(usize)]
    pub enum ArgFileType {
        Csv,
        Tsv,
    }
}

#[derive(StructOpt)]
struct CmdOpt {
    /// Generate a popular data set
    //   blanas: Blanas et al. "Main memory hash join algorithms for multi-core CPUs"
    //   kim: Kim et al. "Sort vs. hash revisited"
    #[structopt(
        long = "data-set",
        raw(possible_values = "&ArgDataSet::variants()", case_insensitive = "true")
    )]
    data_set: Option<ArgDataSet>,

    /// Set the tuple size (bytes)
    #[structopt(
        long = "tuple-bytes",
        default_value = "Bytes8",
        raw(
            possible_values = "&ArgTupleBytes::variants()",
            case_insensitive = "true"
        )
    )]
    tuple_bytes: ArgTupleBytes,

    /// Set the output file type
    #[structopt(
        long = "file-type",
        default_value = "Tsv",
        raw(
            possible_values = "&ArgFileType::variants()",
            case_insensitive = "true"
        )
    )]
    file_type: ArgFileType,

    /// Inner relation output file
    inner_rel_path: PathBuf,

    /// Outer relation output file
    outer_rel_path: PathBuf,

    /// Inner relation size (tuples)
    #[structopt(conflicts_with = "data_set", requires = "outer_rel_tuples")]
    inner_rel_tuples: Option<usize>,

    /// Outer relation size (tuples)
    #[structopt(conflicts_with = "data_set")]
    outer_rel_tuples: Option<usize>,
}

#[derive(Debug, Serialize)]
struct Record<K, V> {
    key: K,
    value: V,
}

fn generate<T, W>(
    _inner_rel_tuples: usize,
    _outer_rel_tuples: usize,
    _inner_writer: &mut csv::Writer<W>,
    _outer_writer: &mut csv::Writer<W>,
) -> Result<()>
where
    T: num_traits::FromPrimitive + Serialize,
    W: Write,
{
    unimplemented! {};
}

type DataGenFn<T> = Box<FnMut(&mut [T], &mut [T]) -> datagen::error::Result<()>>;

fn generate_popular<T, W>(
    data_set: ArgDataSet,
    inner_writer: &mut csv::Writer<W>,
    outer_writer: &mut csv::Writer<W>,
) -> Result<()>
where
    T: Copy + Default + num_traits::FromPrimitive + Serialize,
    W: Write,
{
    let (inner_len, outer_len, mut gen_fn): (usize, usize, DataGenFn<T>) = match data_set {
        ArgDataSet::Blanas => (
            popular::Blanas::primary_key_len(),
            popular::Blanas::foreign_key_len(),
            Box::new(|pk_rel, fk_rel| datagen::popular::Blanas::gen(pk_rel, fk_rel)),
        ),
        ArgDataSet::Kim => (
            popular::Kim::primary_key_len(),
            popular::Kim::foreign_key_len(),
            Box::new(|pk_rel, fk_rel| datagen::popular::Kim::gen(pk_rel, fk_rel)),
        ),
    };

    let mut inner_rel = vec![T::default(); inner_len];
    let mut outer_rel = vec![T::default(); outer_len];
    gen_fn(inner_rel.as_mut_slice(), outer_rel.as_mut_slice())?;

    inner_rel
        .iter()
        .enumerate()
        .map(|(value, key)| {
            let record = Record {
                key,
                value: value + 1,
            };
            inner_writer.serialize(&record)?;
            Ok(())
        })
        .collect::<Result<()>>()?;

    outer_rel
        .iter()
        .enumerate()
        .map(|(value, key)| {
            let record = Record {
                key,
                value: value + 1,
            };
            outer_writer.serialize(&record)?;
            Ok(())
        })
        .collect::<Result<()>>()?;

    Ok(())
}
