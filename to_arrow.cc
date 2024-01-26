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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/interop.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/interop.hpp>
#include <cudf/io/types.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include "arrow/c/abi.h"
#include "arrow/c/bridge.h"
#include "arrow/c/helpers.h"
#include "arrow/memory_pool.h"
#include "arrow/util/span.h"
#include "interop.hpp"

namespace detail {
std::shared_ptr<arrow::DataType> to_arrow_datatype(cudf::type_id id) {
  switch (id) {
    case cudf::type_id::BOOL8:
      return arrow::boolean();
    case cudf::type_id::INT8:
      return arrow::int8();
    case cudf::type_id::INT16:
      return arrow::int16();
    case cudf::type_id::INT32:
      return arrow::int32();
    case cudf::type_id::INT64:
      return arrow::int64();
    case cudf::type_id::UINT8:
      return arrow::uint8();
    case cudf::type_id::UINT16:
      return arrow::uint16();
    case cudf::type_id::UINT32:
      return arrow::uint32();
    case cudf::type_id::UINT64:
      return arrow::uint64();
    case cudf::type_id::FLOAT32:
      return arrow::float32();
    case cudf::type_id::FLOAT64:
      return arrow::float64();
    case cudf::type_id::TIMESTAMP_DAYS:
      return arrow::date32();
    case cudf::type_id::TIMESTAMP_SECONDS:
      return arrow::timestamp(arrow::TimeUnit::SECOND);
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return arrow::timestamp(arrow::TimeUnit::MILLI);
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return arrow::timestamp(arrow::TimeUnit::MICRO);
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return arrow::timestamp(arrow::TimeUnit::NANO);
    case cudf::type_id::DURATION_SECONDS:
      return arrow::duration(arrow::TimeUnit::SECOND);
    case cudf::type_id::DURATION_MILLISECONDS:
      return arrow::duration(arrow::TimeUnit::MILLI);
    case cudf::type_id::DURATION_MICROSECONDS:
      return arrow::duration(arrow::TimeUnit::MICRO);
    case cudf::type_id::DURATION_NANOSECONDS:
      return arrow::duration(arrow::TimeUnit::NANO);
    default:
      CUDF_FAIL("Unsupported type_id conversion to arrow");
  }
}

arrow::Result<struct ArrowArray*> get_null_arr(int sz, arrow::MemoryPool* ar_mr) {
  auto arr = std::make_shared<arrow::NullArray>(sz);
  struct ArrowArray* out;
  ARROW_RETURN_NOT_OK(
      ar_mr->Allocate(sizeof(struct ArrowArray), reinterpret_cast<uint8_t**>(&out)));

  ARROW_RETURN_NOT_OK(arrow::ExportArray(*arr, out));
  return out;
}

struct dispatch_to_arrow_type {
  template <typename T, CUDF_ENABLE_IF(not cudf::is_rep_layout_compatible<T>())>
  std::shared_ptr<arrow::DataType> operator()(cudf::column_view, cudf::type_id,
                                              cudf::io::column_name_info const&) {
    CUDF_FAIL("Unsupported type for to_arrow_schema");
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_rep_layout_compatible<T>())>
  std::shared_ptr<arrow::DataType> operator()(cudf::column_view input, cudf::type_id id,
                                              cudf::io::column_name_info const&) {
    return to_arrow_datatype(id);
  }
};

template <>
std::shared_ptr<arrow::DataType> dispatch_to_arrow_type::operator()<numeric::decimal32>(
    cudf::column_view input, cudf::type_id id, cudf::io::column_name_info const& meta) {
  CUDF_FAIL("Unsupported type for to_arrow_schema");
}

template <>
std::shared_ptr<arrow::DataType> dispatch_to_arrow_type::operator()<numeric::decimal64>(
    cudf::column_view input, cudf::type_id id, cudf::io::column_name_info const& meta) {
  CUDF_FAIL("Unsupported type for to_arrow_schema");
}

template <>
std::shared_ptr<arrow::DataType> dispatch_to_arrow_type::operator()<numeric::decimal128>(
    cudf::column_view input, cudf::type_id id, cudf::io::column_name_info const& meta) {
  using DeviceType = __int128_t;
  auto const max_precision = cudf::detail::max_precision<DeviceType>();
  return arrow::decimal(max_precision, -input.type().scale());
}

template <>
std::shared_ptr<arrow::DataType> dispatch_to_arrow_type::operator()<bool>(
    cudf::column_view input, cudf::type_id id, cudf::io::column_name_info const& meta) {
  return to_arrow_datatype(id);
}

template <>
std::shared_ptr<arrow::DataType> dispatch_to_arrow_type::operator()<cudf::string_view>(
    cudf::column_view input, cudf::type_id id, cudf::io::column_name_info const& meta) {
  return arrow::utf8();
}

template <>
std::shared_ptr<arrow::DataType> dispatch_to_arrow_type::operator()<cudf::dictionary32>(
    cudf::column_view input, cudf::type_id id, cudf::io::column_name_info const& meta) {
  CUDF_FAIL("Unsupported type for to_arrow_schema");
}

template <>
std::shared_ptr<arrow::DataType> dispatch_to_arrow_type::operator()<cudf::list_view>(
    cudf::column_view input, cudf::type_id id, cudf::io::column_name_info const& meta) {
  CUDF_FAIL("Unsupported type for to_arrow_schema");
  // auto child = input.child(0);
  // return arrow::list(cudf::type_dispatcher(child.type(),
  // detail::dispatch_to_arrow_type{},
  //                                          child, child.type().id(),
  //                                          meta.children_meta[0]));
}

template <>
std::shared_ptr<arrow::DataType> dispatch_to_arrow_type::operator()<cudf::struct_view>(
    cudf::column_view input, cudf::type_id id, cudf::io::column_name_info const& meta) {
  CUDF_EXPECTS(meta.children.size() == static_cast<std::size_t>(input.num_children()),
               "Number of field names and number of children don't match\n");

  std::vector<std::shared_ptr<arrow::Field>> fields;
  std::transform(input.child_begin(), input.child_end(), meta.children.cbegin(),
                 std::back_inserter(fields), [](auto const c, auto const meta) {
                   std::shared_ptr<arrow::DataType> typ = arrow::null();
                   if (c.type().id() != cudf::type_id::EMPTY) {
                     typ =
                         cudf::type_dispatcher(c.type(), detail::dispatch_to_arrow_type{},
                                               c, c.type().id(), meta);
                   }

                   return std::make_shared<arrow::Field>(meta.name, typ,
                                                         meta.is_nullable.value_or(true));
                 });
  return std::make_shared<arrow::StructType>(fields);
}

struct dispatch_to_arrow {
  template <typename T, CUDF_ENABLE_IF(not cudf::is_rep_layout_compatible<T>())>
  arrow::Result<struct ArrowArray*> operator()(cudf::column_view, cudf::type_id,
                                               arrow::MemoryPool*,
                                               rmm::cuda_stream_view) {
    CUDF_FAIL("Unsupported type for to_arrow");
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_rep_layout_compatible<T>())>
  arrow::Result<struct ArrowArray*> operator()(cudf::column_view input_view,
                                               cudf::type_id id, arrow::MemoryPool* ar_mr,
                                               rmm::cuda_stream_view stream) {
    struct ArrowArray* out;
    ARROW_RETURN_NOT_OK(
        ar_mr->Allocate(sizeof(struct ArrowArray), reinterpret_cast<uint8_t**>(&out)));
    memset(out, 0, sizeof(struct ArrowArray));

    const void** buffers;
    ARROW_RETURN_NOT_OK(ar_mr->Allocate(
        sizeof(void*) * 2, reinterpret_cast<uint8_t**>(const_cast<void***>(&buffers))));
    buffers[0] = input_view.null_mask();
    buffers[1] = input_view.data<T>();

    *out = (struct ArrowArray){
        .length = input_view.size(),
        .null_count = input_view.null_count(),
        .offset = 0,
        .n_buffers = 2,
        .n_children = 0,
        .buffers = buffers,
        .children = nullptr,
        .dictionary = nullptr,
        .release =
            [](struct ArrowArray* arr) {
              auto* pool = reinterpret_cast<arrow::MemoryPool*>(arr->private_data);
              pool->Free(
                  const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(arr->buffers)),
                  sizeof(void*) * 2);
              ArrowArrayMarkReleased(arr);
            },
        .private_data = reinterpret_cast<void*>(ar_mr),
    };
    return out;
  }
};

template <>
arrow::Result<struct ArrowArray*> dispatch_to_arrow::operator()<numeric::decimal32>(
    cudf::column_view input, cudf::type_id id, arrow::MemoryPool* ar_mr,
    rmm::cuda_stream_view stream) {
  CUDF_FAIL("Unsupported type for to_arrow");
}

template <>
arrow::Result<struct ArrowArray*> dispatch_to_arrow::operator()<numeric::decimal64>(
    cudf::column_view input, cudf::type_id id, arrow::MemoryPool* ar_mr,
    rmm::cuda_stream_view stream) {
  CUDF_FAIL("Unsupported type for to_arrow");
}

template <>
arrow::Result<struct ArrowArray*> dispatch_to_arrow::operator()<numeric::decimal128>(
    cudf::column_view input, cudf::type_id id, arrow::MemoryPool* ar_mr,
    rmm::cuda_stream_view stream) {
  using DeviceType = __int128_t;

  struct ArrowArray* out;
  ARROW_RETURN_NOT_OK(
      ar_mr->Allocate(sizeof(struct ArrowArray), reinterpret_cast<uint8_t**>(&out)));
  memset(out, 0, sizeof(struct ArrowArray));

  const void** buffers;
  ARROW_RETURN_NOT_OK(ar_mr->Allocate(
      sizeof(void*) * 2, reinterpret_cast<uint8_t**>(const_cast<void**>(buffers))));
  buffers[0] = input.null_mask();
  buffers[1] = input.data<DeviceType>();

  *out = (struct ArrowArray){
      .length = input.size(),
      .null_count = input.null_count(),
      .offset = 0,
      .n_buffers = 2,
      .n_children = 0,
      .buffers = buffers,
      .children = nullptr,
      .dictionary = nullptr,
      .release =
          [](struct ArrowArray* arr) {
            auto* pool = reinterpret_cast<arrow::MemoryPool*>(arr->private_data);
            pool->Free(
                const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(*arr->buffers)),
                sizeof(void*) * 2);
            ArrowArrayMarkReleased(arr);
          },
      .private_data = reinterpret_cast<void*>(ar_mr),
  };
  return out;
}

struct boolctx {
  arrow::MemoryPool* pool;
  std::unique_ptr<rmm::device_buffer> buf;
};

template <>
arrow::Result<struct ArrowArray*> dispatch_to_arrow::operator()<bool>(
    cudf::column_view input, cudf::type_id id, arrow::MemoryPool* ar_mr,
    rmm::cuda_stream_view stream) {
  auto bitmask =
      cudf::detail::bools_to_mask(input, stream, rmm::mr::get_current_device_resource());
  struct ArrowArray* out;
  ARROW_RETURN_NOT_OK(
      ar_mr->Allocate(sizeof(struct ArrowArray), reinterpret_cast<uint8_t**>(&out)));
  memset(out, 0, sizeof(struct ArrowArray));

  const void** buffers;
  ARROW_RETURN_NOT_OK(ar_mr->Allocate(
      sizeof(void*) * 2, reinterpret_cast<uint8_t**>(const_cast<void***>(&buffers))));
  buffers[0] = input.null_mask();
  buffers[1] = bitmask.first->data();

  *out = (struct ArrowArray){
      .length = input.size(),
      .null_count = input.null_count(),
      .offset = 0,
      .n_buffers = 2,
      .n_children = 0,
      .buffers = buffers,
      .children = nullptr,
      .dictionary = nullptr,
      .release =
          [](struct ArrowArray* arr) {
            auto* ctx = reinterpret_cast<boolctx*>(arr->private_data);
            ctx->buf.reset();
            ctx->pool->Free(
                const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(arr->buffers)),
                sizeof(void*) * 2);
            delete ctx;
            ArrowArrayMarkReleased(arr);
          },
      .private_data = new boolctx{ar_mr, std::move(bitmask.first)},
  };
  return out;
}

struct empty_string_ctx {
  arrow::MemoryPool* pool;
  rmm::device_scalar<int32_t> zero;
};

template <>
arrow::Result<struct ArrowArray*> dispatch_to_arrow::operator()<cudf::string_view>(
    cudf::column_view input, cudf::type_id id, arrow::MemoryPool* ar_mr,
    rmm::cuda_stream_view stream) {
  struct ArrowArray* out;
  ARROW_RETURN_NOT_OK(
      ar_mr->Allocate(sizeof(struct ArrowArray), reinterpret_cast<uint8_t**>(&out)));
  memset(out, 0, sizeof(struct ArrowArray));

  const void** buffers;
  ARROW_RETURN_NOT_OK(ar_mr->Allocate(
      sizeof(void*) * 3, reinterpret_cast<uint8_t**>(const_cast<void***>(&buffers))));
  if (input.size() == 0) {
    // empty array should have single offset value of 4 bytes
    auto zero = rmm::device_scalar<int32_t>(0, stream);
    buffers[0] = nullptr;
    buffers[1] = reinterpret_cast<const void*>(zero.data());
    buffers[2] = nullptr;

    *out = (struct ArrowArray){
        .length = 0,
        .null_count = 0,
        .offset = 0,
        .n_buffers = 3,
        .n_children = 0,
        .buffers = buffers,
        .children = nullptr,
        .dictionary = nullptr,
        .release =
            [](struct ArrowArray* arr) {
              auto* ctx = reinterpret_cast<empty_string_ctx*>(arr->private_data);
              ctx->pool->Free(
                  const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(arr->buffers)),
                  sizeof(void*) * 3);
              delete ctx;
              ArrowArrayMarkReleased(arr);
            },
        .private_data = new empty_string_ctx{ar_mr, std::move(zero)},
    };
    return out;
  }

  buffers[0] = input.null_mask();
  buffers[1] = input.child(0).data<int32_t>();
  buffers[2] = input.child(1).data<const char>();

  *out = (struct ArrowArray){
      .length = input.size(),
      .null_count = input.null_count(),
      .offset = 0,
      .n_buffers = 3,
      .n_children = 0,
      .buffers = buffers,
      .children = nullptr,
      .dictionary = nullptr,
      .release =
          [](struct ArrowArray* arr) {
            auto* pool = reinterpret_cast<arrow::MemoryPool*>(arr->private_data);
            pool->Free(
                const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(arr->buffers)),
                sizeof(void*) * 3);
            ArrowArrayMarkReleased(arr);
          },
      .private_data = reinterpret_cast<void*>(ar_mr),
  };
  return out;
}

template <>
arrow::Result<struct ArrowArray*> dispatch_to_arrow::operator()<cudf::struct_view>(
    cudf::column_view input, cudf::type_id id, arrow::MemoryPool* ar_mr,
    rmm::cuda_stream_view stream) {
  CUDF_FAIL("Unsupported type for to_arrow");
}

template <>
arrow::Result<struct ArrowArray*> dispatch_to_arrow::operator()<cudf::list_view>(
    cudf::column_view input, cudf::type_id id, arrow::MemoryPool* ar_mr,
    rmm::cuda_stream_view stream) {
  CUDF_FAIL("Unsupported type for to_arrow");
}

template <>
arrow::Result<struct ArrowArray*> dispatch_to_arrow::operator()<cudf::dictionary32>(
    cudf::column_view input, cudf::type_id id, arrow::MemoryPool* ar_mr,
    rmm::cuda_stream_view stream) {
  CUDF_FAIL("Unsupported type for to_arrow");
}

}  // namespace detail

struct dev_arr_ctx {
  std::shared_ptr<cudf::table> tbl;
  std::vector<struct ArrowArray*> children;
  arrow::MemoryPool* pool;
  cudaEvent_t ev;
};

arrow::Status to_arrow_device_arr(std::shared_ptr<cudf::table> input,
                                  struct ArrowDeviceArray* out,
                                  rmm::cuda_stream_view stream,
                                  arrow::MemoryPool* ar_mr) {
  CUDF_EXPECTS(out != nullptr, "must not provide nullptr for ArrowDeviceArray* out var");
  std::memset(out, 0, sizeof(struct ArrowDeviceArray));

  auto input_view = input->view();
  std::vector<struct ArrowArray*> children;
  std::transform(input_view.begin(), input_view.end(), std::back_inserter(children),
                 [&](auto const& c) {
                   return c.type().id() != cudf::type_id::EMPTY
                              ? *cudf::type_dispatcher(c.type(),
                                                       detail::dispatch_to_arrow{}, c,
                                                       c.type().id(), ar_mr, stream)
                              : *detail::get_null_arr(c.size(), ar_mr);
                 });

  dev_arr_ctx* ctx =
      new dev_arr_ctx{std::move(input), std::move(children), ar_mr, nullptr};
  cudaEventCreate(&ctx->ev);

  auto status = cudaEventRecord(ctx->ev, stream);
  if (status != cudaSuccess) {
    for (auto& c : ctx->children) {
      c->release(c);
      ctx->pool->Free(reinterpret_cast<uint8_t*>(c), sizeof(struct ArrowArray));
    }
    delete ctx;
    return arrow::Status::ExecutionError(cudaGetErrorName(status),
                                         cudaGetErrorString(status));
  }

  *out = (struct ArrowDeviceArray){
      .array =
          (struct ArrowArray){
              .length = ctx->tbl->num_rows(),
              .null_count = 0,
              .offset = 0,
              .n_buffers = 1,
              .n_children = ctx->tbl->num_columns(),
              .buffers = (const void**)(malloc(sizeof(void*))),
              .children = ctx->children.data(),
              .dictionary = nullptr,
              .release =
                  [](struct ArrowArray* arr) {
                    auto* ctx = reinterpret_cast<dev_arr_ctx*>(arr->private_data);
                    free(arr->buffers);
                    for (auto& c : ctx->children) {
                      ArrowArrayRelease(c);
                      ctx->pool->Free(reinterpret_cast<uint8_t*>(c),
                                      sizeof(struct ArrowArray));
                    }
                    delete ctx;
                    ArrowArrayMarkReleased(arr);
                  },
              .private_data = ctx,
          },
      .device_id = rmm::get_current_cuda_device().value(),
      .device_type = ARROW_DEVICE_CUDA,
      .sync_event = &ctx->ev,
  };

  out->array.buffers[0] = nullptr;
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::Schema>> to_arrow_schema(
    const cudf::io::table_with_metadata& tbl) {
  std::vector<std::shared_ptr<arrow::Field>> fields;
  auto view = tbl.tbl->view();
  std::transform(view.begin(), view.end(), tbl.metadata.schema_info.begin(),
                 std::back_inserter(fields), [](const auto& c, const auto& meta) {
                   std::shared_ptr<arrow::DataType> typ = arrow::null();
                   if (c.type().id() != cudf::type_id::EMPTY) {
                     typ =
                         cudf::type_dispatcher(c.type(), detail::dispatch_to_arrow_type{},
                                               c, c.type().id(), meta);
                   }
                   return arrow::field(meta.name, typ, meta.is_nullable.value_or(true));
                 });
  return arrow::schema(fields);
}