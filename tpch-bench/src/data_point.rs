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

use crate::error::{ErrorKind, Result};
use crate::types::*;
use serde::Serializer;
use serde_derive::Serialize;

#[derive(Clone, Debug, Default, Serialize)]
pub struct DataPoint {
    pub tpch_query: Option<u32>,
    pub scale_factor: Option<u32>,
    pub selection_variant: Option<ArgSelectionVariant>,
    pub hostname: String,
    pub execution_method: Option<ArgExecutionMethod>,
    #[serde(serialize_with = "serialize_vec")]
    pub device_codename: Option<Vec<String>>,
    pub threads: Option<usize>,
    pub relation_memory_type: Option<ArgMemType>,
    pub relation_memory_location: Option<u16>,
    pub page_type: Option<ArgPageType>,
    pub tuples: Option<usize>,
    pub bytes: Option<usize>,
    pub warm_up: Option<bool>,
    pub ns: Option<f64>,
}

impl DataPoint {
    pub fn new() -> Result<Self> {
        let hostname =
            hostname::get_hostname().ok_or_else(|| ErrorKind::from("Couldn't get hostname"))?;

        let dp = DataPoint {
            hostname,
            ..DataPoint::default()
        };

        Ok(dp)
    }
}

/// Serialize `Option<Vec<T>>` by converting it into a `String`.
///
/// This is necessary because the `csv` crate does not support nesting `Vec`
/// instead of flattening it.
fn serialize_vec<S, T>(option: &Option<Vec<T>>, ser: S) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
    T: ToString,
{
    if let Some(vec) = option {
        let record = vec
            .iter()
            .enumerate()
            .map(|(i, e)| {
                if i == 0 {
                    e.to_string()
                } else {
                    ",".to_owned() + &e.to_string()
                }
            })
            .collect::<String>();
        ser.serialize_str(&record)
    } else {
        ser.serialize_none()
    }
}
