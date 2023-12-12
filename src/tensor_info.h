#ifndef __GST_TENSOE_INFO_H__
#define __GST_TENSOE_INFO_H__

#include <gst/gst.h>
#include <stdlib.h>
#include <sys/types.h>
#if !defined(_WIN32)
#include <strings.h>
#include <stdint.h>
#else
#if (defined(_MSC_VER) && _MSC_VER >= 1600) || (defined(__MSVCRT_VERSION__) && __MSVCRT_VERSION__ >= 1400)
#include <stdint.h>
#elif defined(SCTP_STDINT_INCLUDE)
#include SCTP_STDINT_INCLUDE
#else
#define uint32_t unsigned __int32
#define uint64_t unsigned __int64
#endif
#include <winsock2.h>
#endif

#define NNS_TENSOR_RANK_LIMIT	(16)
#define NNS_TENSOR_SIZE_LIMIT	(16)
#define NNS_TENSOR_SIZE_LIMIT_STR	"16"
#define NNS_TENSOR_SIZE_EXTRA_LIMIT (240)

typedef uint32_t tensor_dim[NNS_TENSOR_RANK_LIMIT];

/**
 * @brief If the given string is NULL, print "(NULL)". Copied from `GST_STR_NULL`
 */
#define _STR_NULL(str) ((str) ? (str) : "(NULL)")

/**
 * @brief Possible data element types of other/tensor.
 */
typedef enum _tensor_type
{
  _TENOR_INT32 = 0,
  _TENOR_UINT32,
  _TENOR_INT16,
  _TENOR_UINT16,
  _TENOR_INT8,
  _TENOR_UINT8,
  _TENOR_FLOAT64,
  _TENOR_FLOAT32,
  _TENOR_INT64,
  _TENOR_UINT64,
  _TENOR_FLOAT16, /**< added with nnstreamer 2.1.1-devel. If you add any operators (e.g., tensor_transform) to float16, it will either be not supported or be too inefficient. */

  _TENOR_END,
} tensor_type;

/**
 * @brief Byte-per-element of each tensor element type.
 */
static const guint tensor_element_size[] = {
  [_TENOR_INT32] = 4,
  [_TENOR_UINT32] = 4,
  [_TENOR_INT16] = 2,
  [_TENOR_UINT16] = 2,
  [_TENOR_INT8] = 1,
  [_TENOR_UINT8] = 1,
  [_TENOR_FLOAT64] = 8,
  [_TENOR_FLOAT32] = 4,
  [_TENOR_INT64] = 8,
  [_TENOR_UINT64] = 8,
  [_TENOR_FLOAT16] = 2,
  [_TENOR_END] = 0,
};

typedef struct
{
  void *data; /**< The instance of tensor data. */
  size_t size; /**< The size of tensor. */
} GstTensorMemory;

typedef enum _tensor_layout
{
  _TENOR_LAYOUT_ANY = 0,     /**< does not care about the data layout */
  _TENOR_LAYOUT_NHWC,        /**< NHWC: channel last layout */
  _TENOR_LAYOUT_NCHW,        /**< NCHW: channel first layout */
  _TENOR_LAYOUT_NONE,        /**< NONE: none of the above defined layouts */
} tensor_layout;

typedef tensor_layout tensors_layout[NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT];

/**
 * @brief Internal data structure for tensor info.
 * @note This must be coherent with api/capi/include/nnstreamer-capi-private.h:ml_tensor_info_s
 */
typedef struct
{
  char *name; /**< Name of each element in the tensor.
                   User must designate this in a few NNFW frameworks (tensorflow)
                   and some (tensorflow-lite) do not need this. */
  tensor_type type; /**< Type of each element in the tensor. User must designate this. */
  tensor_dim dimension; /**< Dimension. We support up to 16th ranks.  */
} GstTensorInfo;

/**
 * @brief Internal meta data exchange format for a other/tensors instance
 * @note This must be coherent with api/capi/include/nnstreamer-capi-private.h:ml_tensors_info_s
 */
typedef struct
{
  unsigned int num_tensors; /**< The number of tensors */
  GstTensorInfo info[NNS_TENSOR_SIZE_LIMIT]; /**< The list of tensor info */
  GstTensorInfo *extra; /**< The list of tensor info for tensors whose idx is larger than NNS_TENSOR_SIZE_LIMIT */
} GstTensorsInfo;

/**
 * @brief Internal data structure for sparse tensor info
 */
typedef struct
{
  uint32_t nnz; /**< the number of "non-zero" elements */
} GstSparseTensorInfo;

/**
 * @brief Data structure to describe a tensor data.
 * This represents the basic information of a memory block for tensor stream.
 *
 * Internally NNStreamer handles a buffer with capability other/tensors-flexible using this information.
 * - version: The version of tensor meta.
 * - type: The type of each element in the tensor. This should be a value of enumeration tensor_type.
 * - dimension: The dimension of tensor. This also denotes the rank of tensor. (e.g., [3:224:224:0] means rank 3.)
 * - format: The data format in the tensor. This should be a value of enumeration tensor_format.
 * - media_type: The media type of tensor. This should be a value of enumeration media_type.
 */
typedef struct
{
  uint32_t magic;
  uint32_t version;
  uint32_t type;
  tensor_dim dimension;
  uint32_t format;
  uint32_t media_type;

  /**
   * @brief Union of the required information for processing each tensor "format".
   */
  union {
    GstSparseTensorInfo sparse_info;
  };

} GstTensorMetaInfo;

void gst_tensor_info_init (GstTensorInfo * info);
void gst_tensor_info_free (GstTensorInfo * info);
gboolean gst_tensor_info_validate (const GstTensorInfo * info);
void gst_tensors_info_init (GstTensorsInfo * info);
void gst_tensors_info_free (GstTensorsInfo * info);
gboolean gst_tensors_info_validate (const GstTensorsInfo * info);
gboolean gst_tensors_info_is_equal (const GstTensorsInfo * i1, const GstTensorsInfo * i2);

GstTensorInfo * gst_tensors_info_get_nth_info (GstTensorsInfo * info, guint nth);
gboolean gst_tensors_info_extra_create (GstTensorsInfo * info);
void gst_tensors_info_extra_free (GstTensorsInfo * info);
guint gst_tensors_info_parse_types_string (GstTensorsInfo * info, const gchar * type_string);
guint gst_tensor_dimension_get_rank (const tensor_dim dim);
gboolean gst_tensor_dimension_is_valid (const tensor_dim dim);
guint gst_tensor_parse_dimension (const gchar * dimstr, tensor_dim dim);
void gst_tensors_layout_init (tensors_layout layout);
void gst_tensors_rank_init (unsigned int ranks[]);

/**
 * raster[ch][12] is the top pixels
 * raster[ch][0] is the bottom pixels
 * raster[ch][height] & 0xF0 is the left-hand side
 * raster[ch][height] & 0x0F is the right-hand size
 */

extern uint8_t rasters[][13];

#endif /* __GST_TENSOE_INFO_H__ */