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
use datagen::relation::{UniformRelation, ZipfRelation};
use serde::ser::Serialize;
use serde_derive::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use structopt::clap::arg_enum;
use structopt::StructOpt;

type Result<T> = std::result::Result<T, Box<std::error::Error>>;

fn main() -> Result<()> {
    // Parse commandline arguments
    let cmd = CmdOpt::from_args();

    match cmd.cmd {
        Command::PkFkJoin(ref join_cmd) => {
            // Convert cmdline args to DataDistribution type
            let distribution = match join_cmd.distribution {
                ArgDistribution::Uniform => DataDistribution::Uniform,
                ArgDistribution::Zipf => DataDistribution::Zipf(
                    join_cmd.zipf_exponent.expect("Zipf exponent not specified"),
                ),
            };

            // Create files for inner and outer relations
            let inner_rel_file = File::create(&join_cmd.inner_rel_path)?;
            let outer_rel_file = File::create(&join_cmd.outer_rel_path)?;

            // Generate relations
            match join_cmd.tuple_bytes {
                ArgTupleBytes::Bytes8 => {
                    let (inner_rel, outer_rel) = if let (Some(inner), Some(outer)) =
                        (join_cmd.inner_rel_tuples, join_cmd.outer_rel_tuples)
                    {
                        generate::<i32>(inner, outer, distribution)?
                    } else if let Some(data_set) = join_cmd.data_set {
                        generate_popular::<i32>(data_set)?
                    } else {
                        unreachable!()
                    };

                    // Write the relations to file
                    write_file(inner_rel.as_slice(), inner_rel_file, join_cmd.file_type)?;
                    write_file(outer_rel.as_slice(), outer_rel_file, join_cmd.file_type)?;
                }
                ArgTupleBytes::Bytes16 => {
                    let (inner_rel, outer_rel) = if let (Some(inner), Some(outer)) =
                        (join_cmd.inner_rel_tuples, join_cmd.outer_rel_tuples)
                    {
                        generate::<i64>(inner, outer, distribution)?
                    } else if let Some(data_set) = join_cmd.data_set {
                        generate_popular::<i64>(data_set)?
                    } else {
                        unreachable!()
                    };

                    // Write the relations to file
                    write_file(inner_rel.as_slice(), inner_rel_file, join_cmd.file_type)?;
                    write_file(outer_rel.as_slice(), outer_rel_file, join_cmd.file_type)?;
                }
            }
        }
    }

    Ok(())
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq)]
    enum ArgDataSet {
        Blanas,
        Kim,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq)]
    enum ArgDistribution {
        Uniform,
        Zipf,
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum DataDistribution {
    Uniform,
    Zipf(f64),
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq)]
    #[repr(usize)]
    enum ArgTupleBytes {
        Bytes8 = 8,
        Bytes16 = 16,
    }
}

arg_enum! {
    #[derive(Copy, Clone, Debug, PartialEq)]
    #[repr(usize)]
    enum ArgFileType {
        Csv,
        Tsv,
    }
}

#[derive(StructOpt)]
struct CmdOpt {
    #[structopt(subcommand)]
    cmd: Command,
}

#[derive(StructOpt)]
enum Command {
    #[structopt(name = "pk-fk-join")]
    PkFkJoin(CmdPkFkJoin),
}

#[derive(StructOpt)]
struct CmdPkFkJoin {
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

    /// Outer relation's data distribution
    #[structopt(
        long = "distribution",
        default_value = "Uniform",
        raw(
            possible_values = "&ArgDistribution::variants()",
            case_insensitive = "true"
        )
    )]
    distribution: ArgDistribution,

    /// Zipf exponent for Zipf-sampled outer relations
    #[structopt(long = "zipf-exponent", raw(required_if = r#""distribution", "Zipf""#))]
    zipf_exponent: Option<f64>,

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

fn generate<T>(
    inner_len: usize,
    outer_len: usize,
    dist: DataDistribution,
) -> Result<(Vec<T>, Vec<T>)>
where
    T: Copy + Default + num_traits::FromPrimitive,
{
    let mut inner_rel = vec![T::default(); inner_len];
    let mut outer_rel = vec![T::default(); outer_len];

    UniformRelation::gen_primary_key(&mut inner_rel)?;

    match dist {
        DataDistribution::Uniform => {
            UniformRelation::gen_foreign_key_from_primary_key(&mut outer_rel, &inner_rel)
        }
        DataDistribution::Zipf(exp) => ZipfRelation::gen_attr(&mut outer_rel, inner_len, exp)?,
    };

    Ok((inner_rel, outer_rel))
}

fn generate_popular<T>(data_set: ArgDataSet) -> Result<(Vec<T>, Vec<T>)>
where
    T: Copy + Default + num_traits::FromPrimitive,
{
    type DataGenFn<T> = Box<FnMut(&mut [T], &mut [T]) -> datagen::error::Result<()>>;

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

    Ok((inner_rel, outer_rel))
}

fn write_file<T, W>(rel: &[T], writer: W, file_type: ArgFileType) -> Result<()>
where
    T: Serialize,
    W: Write,
{
    let mut ser_writer: Box<csv::Writer<_>> = match file_type {
        ArgFileType::Csv => {
            let mut spec = csv::WriterBuilder::new();
            spec.has_headers(true).delimiter(b',');
            Box::new(spec.from_writer(writer))
        }
        ArgFileType::Tsv => {
            let mut spec = csv::WriterBuilder::new();
            spec.has_headers(true).delimiter(b' ');
            Box::new(spec.from_writer(writer))
        }
    };

    rel.iter()
        .enumerate()
        .map(|(value, key)| {
            let record = Record {
                key,
                value: value + 1,
            };
            ser_writer.serialize(&record)?;
            Ok(())
        })
        .collect::<Result<()>>()?;

    Ok(())
}
