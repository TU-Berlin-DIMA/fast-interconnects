/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2019, Clemens Lutz <lutzcle@cml.li>
 * Author: Clemens Lutz <clemens.lutz@dfki.de>
 */

use crossbeam_utils::thread;
use datagen::popular;
use datagen::relation::{KeyAttribute, UniformRelation, ZipfRelation};
use flate2::write::GzEncoder;
use flate2::Compression;
use rand::distributions::uniform::SampleUniform;
use serde::ser::Serialize;
use serde_derive::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use structopt::clap::arg_enum;
use structopt::StructOpt;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

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

            let (inner_rel_writer, outer_rel_writer): (
                Box<dyn Write + Send>,
                Box<dyn Write + Send>,
            ) = if join_cmd.no_compress {
                (Box::new(inner_rel_file), Box::new(outer_rel_file))
            } else {
                (
                    Box::new(GzEncoder::new(inner_rel_file, Compression::default())),
                    Box::new(GzEncoder::new(outer_rel_file, Compression::default())),
                )
            };

            // Generate relations
            match join_cmd.tuple_bytes {
                ArgTupleBytes::Bytes8 => {
                    let (inner_rel, outer_rel) = if let (Some(inner), Some(outer)) =
                        (join_cmd.inner_rel_tuples, join_cmd.outer_rel_tuples)
                    {
                        generate::<i32>(inner, outer, distribution, Some(join_cmd.selectivity))?
                    } else if let Some(data_set) = join_cmd.data_set {
                        generate_popular::<i32>(data_set, Some(join_cmd.selectivity))?
                    } else {
                        unreachable!()
                    };

                    // Write the relations to file
                    thread::scope(|s| {
                        s.spawn(|_| {
                            let pk_timer = Instant::now();
                            write_file(inner_rel.as_slice(), inner_rel_writer, join_cmd.file_type)
                                .expect("Failed to write PK file");
                            let pk_time = Instant::now().duration_since(pk_timer).as_millis();
                            println!("PK write time: {}", pk_time as f64 / 1000.0);
                        });
                        s.spawn(|_| {
                            let fk_timer = Instant::now();
                            write_file(outer_rel.as_slice(), outer_rel_writer, join_cmd.file_type)
                                .expect("Failed to write FK file");
                            let fk_time = Instant::now().duration_since(fk_timer).as_millis();
                            println!("FK write time: {}", fk_time as f64 / 1000.0);
                        });
                    })
                    .expect("Failure inside thread scope");
                }
                ArgTupleBytes::Bytes16 => {
                    let (inner_rel, outer_rel) = if let (Some(inner), Some(outer)) =
                        (join_cmd.inner_rel_tuples, join_cmd.outer_rel_tuples)
                    {
                        generate::<i64>(inner, outer, distribution, Some(join_cmd.selectivity))?
                    } else if let Some(data_set) = join_cmd.data_set {
                        generate_popular::<i64>(data_set, Some(join_cmd.selectivity))?
                    } else {
                        unreachable!()
                    };

                    // Write the relations to file
                    thread::scope(|s| {
                        s.spawn(|_| {
                            let pk_timer = Instant::now();
                            write_file(inner_rel.as_slice(), inner_rel_writer, join_cmd.file_type)
                                .expect("Failed to write PK file");
                            let pk_time = Instant::now().duration_since(pk_timer).as_millis();
                            println!("PK write time: {}", pk_time as f64 / 1000.0);
                        });
                        s.spawn(|_| {
                            let fk_timer = Instant::now();
                            write_file(outer_rel.as_slice(), outer_rel_writer, join_cmd.file_type)
                                .expect("Failed to write FK file");
                            let fk_time = Instant::now().duration_since(fk_timer).as_millis();
                            println!("FK write time: {}", fk_time as f64 / 1000.0);
                        });
                    })
                    .expect("Failure inside thread scope");
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

    /// Enable gzip compression
    #[structopt(long = "no-compress")]
    no_compress: bool,

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

    /// Selectivity of the join, in percent
    #[structopt(
        long = "selectivity",
        default_value = "100",
        raw(validator = "is_percent")
    )]
    selectivity: u32,

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

fn is_percent(x: String) -> std::result::Result<(), String> {
    x.parse::<i32>()
        .map_err(|_| {
            String::from(
                "Failed to parse integer. The value must be a percentage between [0, 100].",
            )
        })
        .and_then(|x| {
            if 0 <= x && x <= 100 {
                Ok(())
            } else {
                Err(String::from(
                    "The value must be a percentage between [0, 100].",
                ))
            }
        })
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
    selectivity: Option<u32>,
) -> Result<(Vec<T>, Vec<T>)>
where
    T: Copy + Default + Send + KeyAttribute + num_traits::FromPrimitive + SampleUniform,
{
    let mut inner_rel = vec![T::default(); inner_len];
    let mut outer_rel = vec![T::default(); outer_len];

    let pk_timer = Instant::now();
    UniformRelation::gen_primary_key_par(&mut inner_rel, selectivity)?;
    let pk_time = Instant::now().duration_since(pk_timer).as_millis();
    println!("PK gen time: {}", pk_time as f64 / 1000.0);

    let fk_timer = Instant::now();
    match dist {
        DataDistribution::Uniform => {
            UniformRelation::gen_attr_par(&mut outer_rel, 1..=inner_rel.len())?
        }
        DataDistribution::Zipf(exp) => ZipfRelation::gen_attr_par(&mut outer_rel, inner_len, exp)?,
    };
    let fk_time = Instant::now().duration_since(fk_timer).as_millis();
    println!("FK gen time: {}", fk_time as f64 / 1000.0);

    Ok((inner_rel, outer_rel))
}

fn generate_popular<T: Send + KeyAttribute>(
    data_set: ArgDataSet,
    selectivity: Option<u32>,
) -> Result<(Vec<T>, Vec<T>)>
where
    T: Copy + Default + num_traits::FromPrimitive,
{
    type DataGenFn<T> = Box<dyn FnMut(&mut [T], &mut [T]) -> datagen::error::Result<()>>;

    let (inner_len, outer_len, mut gen_fn): (usize, usize, DataGenFn<T>) = match data_set {
        ArgDataSet::Blanas => (
            popular::Blanas::primary_key_len(),
            popular::Blanas::foreign_key_len(),
            Box::new(move |pk_rel, fk_rel| {
                datagen::popular::Blanas::gen(pk_rel, fk_rel, selectivity)
            }),
        ),
        ArgDataSet::Kim => (
            popular::Kim::primary_key_len(),
            popular::Kim::foreign_key_len(),
            Box::new(move |pk_rel, fk_rel| datagen::popular::Kim::gen(pk_rel, fk_rel, selectivity)),
        ),
    };

    let mut inner_rel = vec![T::default(); inner_len];
    let mut outer_rel = vec![T::default(); outer_len];
    gen_fn(inner_rel.as_mut_slice(), outer_rel.as_mut_slice())?;

    Ok((inner_rel, outer_rel))
}

fn write_file(rel: &[impl Serialize], writer: impl Write, file_type: ArgFileType) -> Result<()> {
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
