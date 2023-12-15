/*
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
 * Copyright (C) 2020 Niels De Graef <niels.degraef@gmail.com>
 * Copyright (C) 2023 qian <<user@hostname.org>>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Alternatively, the contents of this file may be used under the
 * GNU Lesser General Public License Version 2.1 (the "LGPL"), in
 * which case the following provisions apply instead of the ones
 * mentioned above:
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef __GST_SSCMAYOLOV5_H__
#define __GST_SSCMAYOLOV5_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video-info.h>
#include "tensor_info.h"
#include <net.h>

G_BEGIN_DECLS

#ifndef UNUSED
#define UNUSED(expr) do { (void)(expr); } while (0)
#endif

#define GST_TYPE_SSCMAYOLOV5 \
  (gst_sscma_yolov5_get_type())
#define GST_SWIFT_YOLOV5(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_SSCMAYOLOV5,GstSscmaYolov5))
#define GST_SWIFT_YOLOV5_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_SSCMAYOLOV5,GstSscmaYolov5Class))
#define GST_IS_SWIFT_YOLOV5(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_SSCMAYOLOV5))
#define GST_IS_SWIFT_YOLOV5_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_SSCMAYOLOV5))
#define GST_SWIFT_YOLOV5_CAST(obj)  ((GstSscmaYolov5 *)(obj))

/**
 * @brief Caps string for supported video format
 */
#define VIDEO_CAPS_STR \
    GST_VIDEO_CAPS_MAKE ("{ RGB}") \
    ", interlace-mode = (string) progressive"

/**
 * @brief Caps string for text input
 */
#define TEXT_CAPS_STR "text/x-json"

#define append_video_caps_template(caps) \
    gst_caps_append (caps, gst_caps_from_string (VIDEO_CAPS_STR))

#define append_text_caps_template(caps) \
    gst_caps_append (caps, gst_caps_from_string (TEXT_CAPS_STR))

#define DETECTION_NUM_INFO 5
#define PIXEL_VALUE                             (0xFF) 

/** @brief Represents a detect object */
typedef struct
{
  int valid;
  int class_id;
  int x;
  int y;
  int width;
  int height;
  gfloat prob;

  int tracking_id;
} detectedObject;

typedef struct _GstSscmaYolov5 GstSscmaYolov5;
typedef struct _GstSscmaYolov5Class GstSscmaYolov5Class;

/**
 * @brief GstSscmaYolov5Class inherits GstElementClass.
 *
 * Referring another child (sibiling), GstVideoFilter (abstract class) and
 * its child (concrete class) GstVideoConverter.
 * Note that GstSscmaYolov5Class is a concrete class; thus we need to look at both.
 */
struct _GstSscmaYolov5Class
{
  GstElementClass parent_class;   /**< Inherits GstElementClass */
};

/**
 * @brief GstMode's properties for NN framework (internal data structure)
 *
 * Because custom filters of sscma_yolov5 may need to access internal data
 * of GstSscmaYolov5, we define this data structure here.
 */
typedef struct _GstSscmaYolov5Properties
{
  const char **model_files; /**< Filepath to the model file (as an argument for NNFW). char instead of gchar for non-glib custom plugins */
  int num_models; /**< number of model files. Some frameworks need multiple model files to initialize the graph (caffe, caffe2) */
  int num_threads; /**< number of threads for NNFW */
  bool is_output_scaled; /**< TRUE if output tensor is scaled */

  char **labels; /**< The list of loaded labels. Null if not loaded */
  uint total_labels; /**< The number of loaded labels */
  uint max_word_length; /**< The max size of labels */

  float threshold[3]; /**< The threshold for detection */
  
  GstTensorsInfo input_meta; /**< configured input tensor info */
  tensors_layout input_layout; /**< data layout info provided as a property to sscma_yolov5 for the input, defaults to _NNS_LAYOUT_ANY for all the tensors */
  unsigned int input_ranks[NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT];  /**< the rank list of input tensors, it is calculated based on the dimension string. */

  GstTensorsInfo output_meta; /**< configured output tensor info */
  tensors_layout output_layout; /**< data layout info provided as a property to sscma_yolov5 for the output, defaults to _NNS_LAYOUT_ANY for all the tensors */
  unsigned int output_ranks[NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT];  /**< the rank list of output tensors, it is calculated based on the dimension string. */
} GstSscmaYolov5Properties;

struct _GstSscmaYolov5
{
  GstElement element;

  GstPad *sinkpad, *srcpad;

  ncnn::Net net; /**< NNFW's net object */

  int rate_n; /**< framerate is in fraction, which is numerator/denominator */
  int rate_d; /**< framerate is in fraction, which is numerator/denominator */
  GstTensorsInfo input_info; /**< input tensor info */

  GstSscmaYolov5Properties prop; /**< NNFW plugin's properties */
};

G_END_DECLS

#endif /* __GST_SSCMAYOLOV5_H__ */
