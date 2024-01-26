// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include "arrow/c/abi.h"
#include "arrow/memory_pool.h"

arrow::Status to_arrow_device_arr(
    std::shared_ptr<cudf::table> input, struct ArrowDeviceArray* out,
    rmm::cuda_stream_view stream = cudf::get_default_stream(),
    arrow::MemoryPool* ar_mr = arrow::default_memory_pool());

arrow::Result<std::shared_ptr<arrow::Schema>> to_arrow_schema(
    const cudf::io::table_with_metadata& tbl);