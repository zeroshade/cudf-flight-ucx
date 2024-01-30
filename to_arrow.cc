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

void get_null_arr(int sz, struct ArrowArray* out) {
  auto arr = std::make_shared<arrow::NullArray>(sz);
  ARROW_UNUSED(arrow::ExportArray(*arr, out));
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
  cudf::dictionary_column_view dview{input};

  return arrow::dictionary(to_arrow_datatype(dview.indices().type().id()),
                           to_arrow_datatype(dview.keys_type().id()));
}

template <>
std::shared_ptr<arrow::DataType> dispatch_to_arrow_type::operator()<cudf::struct_view>(
    cudf::column_view input, cudf::type_id id, cudf::io::column_name_info const& meta);

template <>
std::shared_ptr<arrow::DataType> dispatch_to_arrow_type::operator()<cudf::list_view>(
    cudf::column_view input, cudf::type_id id, cudf::io::column_name_info const& meta) {
  auto child = input.child(0);
  return arrow::list(cudf::type_dispatcher(child.type(), detail::dispatch_to_arrow_type{},
                                           child, child.type().id(), meta.children[0]));
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

namespace {
struct ExportedPrivateData {
  std::array<const void*, 3> buffers_;
  struct ArrowArray dictionary_;
  std::vector<struct ArrowArray> children_;
  std::vector<struct ArrowArray*> child_ptrs_;

  ExportedPrivateData() = default;
  ARROW_DEFAULT_MOVE_AND_ASSIGN(ExportedPrivateData);
  ARROW_DISALLOW_COPY_AND_ASSIGN(ExportedPrivateData);
};

}  // namespace

struct dispatch_to_arrow {
  template <typename T, CUDF_ENABLE_IF(not cudf::is_rep_layout_compatible<T>())>
  void operator()(cudf::column_view, cudf::type_id, struct ArrowArray* out,
                  rmm::cuda_stream_view) {
    CUDF_FAIL("Unsupported type for to_arrow");
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_rep_layout_compatible<T>())>
  void operator()(cudf::column_view input_view, cudf::type_id, struct ArrowArray* out,
                  rmm::cuda_stream_view) {
    std::memset(out, 0, sizeof(struct ArrowArray));

    const void** buffers = (const void**)(malloc(sizeof(void*) * 2));
    buffers[0] = input_view.null_mask();
    buffers[1] = input_view.head<T>();

    *out = (struct ArrowArray){
        .length = input_view.size(),
        .null_count = input_view.null_count(),
        .offset = input_view.offset(),
        .n_buffers = 2,
        .n_children = 0,
        .buffers = buffers,
        .children = nullptr,
        .dictionary = nullptr,
        .release =
            [](struct ArrowArray* arr) {
              free(arr->buffers);
              ArrowArrayMarkReleased(arr);
            },
        .private_data = nullptr,
    };
  }
};

template <>
void dispatch_to_arrow::operator()<numeric::decimal32>(cudf::column_view, cudf::type_id,
                                                       struct ArrowArray* out,
                                                       rmm::cuda_stream_view) {
  CUDF_FAIL("Unsupported type for to_arrow");
}

template <>
void dispatch_to_arrow::operator()<numeric::decimal64>(cudf::column_view, cudf::type_id,
                                                       struct ArrowArray* out,
                                                       rmm::cuda_stream_view) {
  CUDF_FAIL("Unsupported type for to_arrow");
}

template <>
void dispatch_to_arrow::operator()<numeric::decimal128>(cudf::column_view input,
                                                        cudf::type_id,
                                                        struct ArrowArray* out,
                                                        rmm::cuda_stream_view) {
  using DeviceType = __int128_t;

  std::memset(out, 0, sizeof(struct ArrowArray));

  const void** buffers = (const void**)(malloc(sizeof(void*) * 2));
  buffers[0] = input.null_mask();
  buffers[1] = input.head<DeviceType>();

  *out = (struct ArrowArray){
      .length = input.size(),
      .null_count = input.null_count(),
      .offset = input.offset(),
      .n_buffers = 2,
      .n_children = 0,
      .buffers = buffers,
      .children = nullptr,
      .dictionary = nullptr,
      .release =
          [](struct ArrowArray* arr) {
            free(arr->buffers);
            ArrowArrayMarkReleased(arr);
          },
      .private_data = nullptr,
  };
}

struct boolctx : public ExportedPrivateData {
  std::unique_ptr<rmm::device_buffer> buf;
};

template <>
void dispatch_to_arrow::operator()<bool>(cudf::column_view input, cudf::type_id id,
                                         struct ArrowArray* out,
                                         rmm::cuda_stream_view stream) {
  std::memset(out, 0, sizeof(struct ArrowArray));

  cudf::column_view view_without_offset = input;
  if (input.offset()) {
    view_without_offset =
        cudf::column_view{input.type(), input.size() + input.offset(), input.head(),
                          input.null_mask(), input.null_count()};
  }
  auto bitmask = cudf::detail::bools_to_mask(view_without_offset, stream,
                                             rmm::mr::get_current_device_resource());

  auto* pdata = new boolctx();
  pdata->buf = std::move(bitmask.first);
  pdata->buffers_[0] = input.null_mask();
  pdata->buffers_[1] = bitmask.first->data();

  *out = (struct ArrowArray){
      .length = input.size(),
      .null_count = input.null_count(),
      .offset = input.offset(),
      .n_buffers = 2,
      .n_children = 0,
      .buffers = pdata->buffers_.data(),
      .children = nullptr,
      .dictionary = nullptr,
      .release =
          [](struct ArrowArray* arr) {
            auto* ctx = reinterpret_cast<boolctx*>(arr->private_data);
            delete ctx;
            ArrowArrayMarkReleased(arr);
          },
      .private_data = pdata,
  };
}

struct empty_string_ctx : public ExportedPrivateData {
  rmm::device_scalar<int32_t> zero;

  empty_string_ctx(rmm::device_scalar<int32_t>&& z) : zero{std::move(z)} {}
};

template <>
void dispatch_to_arrow::operator()<cudf::string_view>(cudf::column_view input,
                                                      cudf::type_id id,
                                                      struct ArrowArray* out,
                                                      rmm::cuda_stream_view stream) {
  std::memset(out, 0, sizeof(struct ArrowArray));

  if (input.size() == 0) {
    // empty array should have single offset value of 4 bytes
    auto* pdata = new empty_string_ctx(rmm::device_scalar<int32_t>(0, stream));
    pdata->buffers_[0] = nullptr;
    pdata->buffers_[1] = reinterpret_cast<const void*>(pdata->zero.data());
    pdata->buffers_[2] = nullptr;

    *out = (struct ArrowArray){
        .length = 0,
        .null_count = 0,
        .offset = 0,
        .n_buffers = 3,
        .n_children = 0,
        .buffers = pdata->buffers_.data(),
        .children = nullptr,
        .dictionary = nullptr,
        .release =
            [](struct ArrowArray* arr) {
              auto* ctx = reinterpret_cast<empty_string_ctx*>(arr->private_data);
              delete ctx;
              ArrowArrayMarkReleased(arr);
            },
        .private_data = pdata,
    };
    return;
  }

  cudf::strings_column_view sview{input};
  const void** buffers = (const void**)(malloc(sizeof(void*) * 3));
  buffers[0] = sview.null_mask();
  buffers[1] = sview.offsets().head<int32_t>();
  buffers[2] = sview.chars().head<const char>();

  *out = (struct ArrowArray){
      .length = sview.size(),
      .null_count = sview.null_count(),
      .offset = sview.offset(),
      .n_buffers = 3,
      .n_children = 0,
      .buffers = buffers,
      .children = nullptr,
      .dictionary = nullptr,
      .release =
          [](struct ArrowArray* arr) {
            free(arr->buffers);
            ArrowArrayMarkReleased(arr);
          },
      .private_data = nullptr,
  };
}

template <>
void dispatch_to_arrow::operator()<cudf::list_view>(cudf::column_view input,
                                                    cudf::type_id id,
                                                    struct ArrowArray* out,
                                                    rmm::cuda_stream_view stream);

template <>
void dispatch_to_arrow::operator()<cudf::dictionary32>(cudf::column_view input,
                                                       cudf::type_id id,
                                                       struct ArrowArray* out,
                                                       rmm::cuda_stream_view stream);

template <>
void dispatch_to_arrow::operator()<cudf::struct_view>(cudf::column_view input,
                                                      cudf::type_id id,
                                                      struct ArrowArray* out,
                                                      rmm::cuda_stream_view stream) {
  std::memset(out, 0, sizeof(struct ArrowArray));

  auto* pdata = new ExportedPrivateData();
  pdata->buffers_[0] = input.null_mask();

  pdata->children_.resize(input.num_children());
  pdata->child_ptrs_.reserve(input.num_children());
  cudf::structs_column_view sview{input};
  for (auto i = 0; i < sview.num_children(); i++) {
    auto* child = &pdata->children_[i];
    auto c = sview.child(i);
    c.type().id() != cudf::type_id::EMPTY
        ? cudf::type_dispatcher(c.type(), detail::dispatch_to_arrow{}, c, c.type().id(),
                                child, stream)
        : detail::get_null_arr(c.size(), child);
    pdata->child_ptrs_.push_back(child);
  }

  *out = (struct ArrowArray){
      .length = sview.size(),
      .null_count = sview.null_count(),
      .offset = sview.offset(),
      .n_buffers = 1,
      .n_children = sview.num_children(),
      .buffers = pdata->buffers_.data(),
      .children = pdata->child_ptrs_.data(),
      .dictionary = nullptr,
      .release =
          [](struct ArrowArray* arr) {
            auto* ctx = reinterpret_cast<ExportedPrivateData*>(arr->private_data);
            for (auto& c : ctx->child_ptrs_) {
              ArrowArrayRelease(c);
            }
            delete ctx;
            ArrowArrayMarkReleased(arr);
          },
      .private_data = pdata,
  };
}

template <>
void dispatch_to_arrow::operator()<cudf::list_view>(cudf::column_view input,
                                                    cudf::type_id id,
                                                    struct ArrowArray* out,
                                                    rmm::cuda_stream_view stream) {
  std::memset(out, 0, sizeof(struct ArrowArray));

  cudf::lists_column_view lview{input};
  auto* pdata = new ExportedPrivateData();
  auto sliced_offsets = cudf::slice(
      lview.offsets(), {input.offset(), input.offset() + input.size() + 1}, stream)[0];

  pdata->buffers_[0] = input.null_mask();
  pdata->buffers_[1] = sliced_offsets.head<const uint8_t>();
  pdata->children_.resize(1);
  pdata->child_ptrs_.resize(1);

  auto c = lview.child();
  c.type().id() != cudf::type_id::EMPTY
      ? cudf::type_dispatcher(c.type(), detail::dispatch_to_arrow{}, c, c.type().id(),
                              &pdata->children_[0], stream)
      : detail::get_null_arr(c.size(), &pdata->children_[0]);
  pdata->child_ptrs_[0] = &pdata->children_[0];

  *out = (struct ArrowArray){
      .length = lview.size(),
      .null_count = lview.null_count(),
      .offset = lview.offset(),
      .n_buffers = 2,
      .n_children = 1,
      .buffers = pdata->buffers_.data(),
      .children = pdata->child_ptrs_.data(),
      .dictionary = nullptr,
      .release =
          [](struct ArrowArray* arr) {
            auto* ctx = reinterpret_cast<ExportedPrivateData*>(arr->private_data);
            ArrowArrayRelease(ctx->child_ptrs_[0]);
            delete ctx;
            ArrowArrayMarkReleased(arr);
          },
      .private_data = pdata,
  };
}

template <>
void dispatch_to_arrow::operator()<cudf::dictionary32>(cudf::column_view input,
                                                       cudf::type_id id,
                                                       struct ArrowArray* out,
                                                       rmm::cuda_stream_view stream) {
  std::memset(out, 0, sizeof(struct ArrowArray));

  cudf::dictionary_column_view dview{input};
  auto indices_view = dview.indices();

  auto* pdata = new ExportedPrivateData();
  pdata->buffers_[0] = dview.null_mask();
  pdata->buffers_[1] = dview.indices().head<const uint8_t>();

  auto keys = dview.keys();
  cudf::type_dispatcher(keys.type(), detail::dispatch_to_arrow{}, keys, keys.type().id(),
                        &pdata->dictionary_, stream);

  *out = (struct ArrowArray){
      .length = dview.size(),
      .null_count = dview.null_count(),
      .offset = dview.offset(),
      .n_buffers = 2,
      .n_children = 0,
      .buffers = pdata->buffers_.data(),
      .children = nullptr,
      .dictionary = &pdata->dictionary_,
      .release =
          [](struct ArrowArray* arr) {
            auto* ctx = reinterpret_cast<ExportedPrivateData*>(arr->private_data);
            ArrowArrayRelease(&ctx->dictionary_);
            delete ctx;
            ArrowArrayMarkReleased(arr);
          },
      .private_data = pdata,
  };
}

}  // namespace detail

struct dev_arr_ctx : public detail::ExportedPrivateData {
  std::shared_ptr<cudf::table> tbl;
  cudaEvent_t ev;
};

arrow::Status to_arrow_device_arr(std::shared_ptr<cudf::table> input,
                                  struct ArrowDeviceArray* out,
                                  rmm::cuda_stream_view stream) {
  CUDF_EXPECTS(out != nullptr, "must not provide nullptr for ArrowDeviceArray* out var");
  std::memset(out, 0, sizeof(struct ArrowDeviceArray));

  auto input_view = input->view();
  auto* pdata = new dev_arr_ctx();

  pdata->child_ptrs_.reserve(input->num_columns());
  pdata->children_.resize(input->num_columns());
  std::transform(input_view.begin(), input_view.end(), pdata->children_.begin(),
                 std::back_inserter(pdata->child_ptrs_), [&](auto const& c, auto& child) {
                   c.type().id() != cudf::type_id::EMPTY
                       ? cudf::type_dispatcher(c.type(), detail::dispatch_to_arrow{}, c,
                                               c.type().id(), &child, stream)
                       : detail::get_null_arr(c.size(), &child);
                   return &child;
                 });

  pdata->tbl = std::move(input);
  cudaEventCreate(&pdata->ev);

  auto status = cudaEventRecord(pdata->ev, stream);
  if (status != cudaSuccess) {
    for (auto& c : pdata->children_) {
      c.release(&c);
    }
    delete pdata;
    return arrow::Status::ExecutionError(cudaGetErrorName(status),
                                         cudaGetErrorString(status));
  }

  *out = (struct ArrowDeviceArray){
      .array =
          (struct ArrowArray){
              .length = pdata->tbl->num_rows(),
              .null_count = 0,
              .offset = 0,
              .n_buffers = 1,
              .n_children = pdata->tbl->num_columns(),
              .buffers = pdata->buffers_.data(),
              .children = pdata->child_ptrs_.data(),
              .dictionary = nullptr,
              .release =
                  [](struct ArrowArray* arr) {
                    auto* ctx = reinterpret_cast<dev_arr_ctx*>(arr->private_data);
                    for (auto& c : ctx->child_ptrs_) {
                      ArrowArrayRelease(c);
                    }
                    cudaEventDestroy(ctx->ev);
                    delete ctx;
                    ArrowArrayMarkReleased(arr);
                  },
              .private_data = pdata,
          },
      .device_id = rmm::get_current_cuda_device().value(),
      .device_type = ARROW_DEVICE_CUDA,
      .sync_event = &pdata->ev,
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