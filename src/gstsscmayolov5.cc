/*
 * GStreamer
 * Copyright (C) 2005 Thomas Vander Stichele <thomas@apestaart.org>
 * Copyright (C) 2005 Ronald S. Bultje <rbultje@ronald.bitfreak.net>
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

/**
 * SECTION:element-sscmayolov5
 *
 * FIXME:Describe sscmayolov5 here.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! sscmayolov5 ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif
#include <string.h>
#include <gst/gst.h>
#include <gst/base/base.h>
#include <gst/controller/controller.h>

#include "gstsscmayolov5.h"
#include "tensor_info.h"
#include <net.h>

GST_DEBUG_CATEGORY_STATIC (gst_sscma_yolov5_debug);
#define GST_CAT_DEFAULT gst_sscma_yolov5_debug

/* Filter signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_SILENT,
  PROP_INPUT,
  PROP_INPUTFORMAT,
  PROP_INPUTTYPE,
  PROP_OUTPUT,
  PROP_OUTPUTTYPE,
  PROP_MODEL,
  PROP_MODE_LABELS,
  PROP_OUTPUTRANKS
};

/* the capabilities of the outputs.
 *
 * describe the real formats here.
 */
static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("ANY")
    );

static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("ANY")
    );

#define gst_sscma_yolov5_parent_class parent_class
G_DEFINE_TYPE (GstSscmaYolov5, gst_sscma_yolov5, GST_TYPE_ELEMENT);

GST_ELEMENT_REGISTER_DEFINE (sscma_yolov5, "sscma_yolov5", GST_RANK_NONE,
    GST_TYPE_SSCMAYOLOV5);

ncnn::Net net;

static void gst_sscma_yolov5_set_property (GObject * object,
    guint prop_id, const GValue * value, GParamSpec * pspec);
static void gst_sscma_yolov5_get_property (GObject * object,
    guint prop_id, GValue * value, GParamSpec * pspec);
static void gst_sscma_yolov5_finalize (GObject * object);

static gboolean gst_sscma_yolov5_sink_event (GstPad * pad,
    GstObject * parent, GstEvent * event);
static gboolean gst_sscma_yolov5_sink_query (GstPad * pad,
    GstObject * parent, GstQuery * query);
static gboolean gst_sscma_yolov5_src_query (GstPad * pad,
    GstObject * parent, GstQuery * query);
static GstFlowReturn gst_sscma_yolov5_chain (GstPad * pad,
    GstObject * parent, GstBuffer * buf);

static GstCaps * gst_sscma_yolov5_query_caps (GstSscmaYolov5 * self, GstPad * pad,
    GstCaps * filter);
static gboolean gst_sscma_yolov5_parse_caps (GstSscmaYolov5 * self,
    const GstCaps * caps);
static gboolean gst_sscma_yolov5_update_caps (GstSscmaYolov5 * self);

static void nms (GArray * results, gfloat threshold);
static void draw (GstMapInfo * out_info, GstSscmaYolov5Properties *prop, GArray * results);
/* initialize the sscmayolov5's class */
static void
gst_sscma_yolov5_class_init (GstSscmaYolov5Class * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstPadTemplate *pad_template;
  GstCaps *pad_caps;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  gobject_class->set_property = gst_sscma_yolov5_set_property;
  gobject_class->get_property = gst_sscma_yolov5_get_property;
  gobject_class->finalize = gst_sscma_yolov5_finalize;

  g_object_class_install_property (gobject_class, PROP_MODEL,
      g_param_spec_string ("model", "Model filepath",
          "File path to the model file. Separated with ',' in case of multiple model files(like caffe2)",
          "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_OUTPUT,
      g_param_spec_string ("output", "Output dimension",
          "Output tensor dimension from inner array (Max rank #NNS_TENSOR_RANK_LIMIT)",
          "", G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_OUTPUTTYPE,
      g_param_spec_string ("outputtype", "Output tensor element type",
          "Type of each element of the output tensor ?", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_OUTPUTRANKS,
      g_param_spec_string ("outputranks", "Rank of Out Tensor",
          "The Rank of the Out Tensor, which is separated with ',' in case of multiple Tensors",
          "", G_PARAM_READABLE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MODE_LABELS,
      g_param_spec_string ("labels", "Labels file",
          "Configure the Labels file path.", "",
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_SILENT,
      g_param_spec_boolean ("silent", "Silent", "Produce verbose output ?",
          FALSE, G_PARAM_READWRITE));

  gst_element_class_set_static_metadata (gstelement_class,
      "SscmaYolov5",
      "FIXME:Generic",
      "swift yolov5", "qian <<ruiqian.tang@seeed.org>>");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));
  /* set sink pad template */
  // pad_caps = gst_caps_new_empty ();
  // append_video_caps_template (pad_caps);

  // pad_template = gst_pad_template_new ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
  //     pad_caps);
  // gst_element_class_add_pad_template (gstelement_class, pad_template);
  // gst_caps_unref (pad_caps);
}

/* initialize the new element
 * instantiate pads and add them to element
 * set pad callback functions
 * initialize instance structure
 */
static void
gst_sscma_yolov5_init (GstSscmaYolov5 * self)
{
  GstSscmaYolov5Properties *prop = &self->prop;

  /** setup sink pad */
  self->sinkpad =
      gst_pad_new_from_template (gst_element_class_get_pad_template
      (GST_ELEMENT_GET_CLASS (self), "sink"), "sink");
  gst_pad_set_event_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_sscma_yolov5_sink_event));
  gst_pad_set_query_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_sscma_yolov5_sink_query));
  gst_pad_set_chain_function (self->sinkpad,
      GST_DEBUG_FUNCPTR (gst_sscma_yolov5_chain));
  GST_PAD_SET_PROXY_CAPS (self->sinkpad);
  gst_element_add_pad (GST_ELEMENT (self), self->sinkpad);

  /** setup src pad */
  self->srcpad =
      gst_pad_new_from_template (gst_element_class_get_pad_template
      (GST_ELEMENT_GET_CLASS (self), "src"), "src");
  gst_pad_set_query_function (self->srcpad,
      GST_DEBUG_FUNCPTR (gst_sscma_yolov5_src_query));
  GST_PAD_SET_PROXY_CAPS (self->srcpad);
  gst_element_add_pad (GST_ELEMENT (self), self->srcpad);

  /* init null */
  memset (prop, 0, sizeof (GstSscmaYolov5Properties));

  gst_tensors_info_init (&prop->input_meta);
  gst_tensors_layout_init (prop->input_layout);
  gst_tensors_rank_init (prop->input_ranks);
}

/**
 * @brief Function to finalize instance.
 */
static void
gst_sscma_yolov5_finalize (GObject * object)
{
  GstSscmaYolov5 *self;
  GstSscmaYolov5Properties *prop;

  self = GST_SWIFT_YOLOV5 (object);
  prop = &self->prop;

  // gst_tensor_filter_common_close_fw (prop);
  gst_tensors_info_free (&prop->input_meta);
  // 释放 self->net 内存
  self->net.clear();
  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/** @brief Handle "PROP_MODEL" for set-property */
static gint
_gtfc_setprop_MODEL (GstSscmaYolov5 * priv,
    GstSscmaYolov5Properties * prop, const GValue * value)
{
  gint status = 0;
  const gchar *model_files = g_value_get_string (value);
  GstSscmaYolov5Properties _prop;

  if (!model_files) {
    g_print ("Invalid model provided to the tensor-filter.");
    return 0;
  }
  _prop.model_files = NULL;

  g_strfreev (prop->model_files);

  if (model_files) {
    prop->model_files = (const gchar **) g_strsplit_set (model_files, ",", -1);
    prop->num_models = g_strv_length ((gchar **) prop->model_files);
  } else {
    prop->model_files = NULL;
    prop->num_models = 0;
  }
  return 0;
}

/**
 * @brief Load label file into the internal data
 * @param[in/out] l The given ImageLabelData struct.
 */
void
loadImageLabels (const char *label_path, GstSscmaYolov5Properties * prop)
{
  GError *err = NULL;
  gchar **_labels;
  gchar *contents = NULL;
  gsize len;
  guint i;

  // init labels
  if (prop->labels) {
    for (i = 0; i < prop->total_labels; i++)
      g_free (prop->labels[i]);
    g_free (prop->labels);
  }
  prop->labels = NULL;
  prop->total_labels = 0;
  prop->max_word_length = 0;

  /* Read file contents */
  if (!g_file_get_contents (label_path, &contents, &len, &err)) {
    g_print ("Unable to read file %s with error %s.", label_path, err->message);
    g_clear_error (&err);
    return;
  }

  if (contents[len - 1] == '\n')
    contents[len - 1] = '\0';

  _labels = g_strsplit (contents, "\n", -1);
  prop->total_labels = g_strv_length (_labels);
  prop->labels = g_new0 (char *, prop->total_labels);
  if (prop->labels == NULL) {
    g_print ("Failed to allocate memory for label data.");
    prop->total_labels = 0;
    goto error;
  }

  for (i = 0; i < prop->total_labels; i++) {
    prop->labels[i] = g_strdup (_labels[i]);

    len = strlen (_labels[i]);
    if (len > prop->max_word_length) {
      prop->max_word_length = len;
    }
  }

error:
  g_strfreev (_labels);
  g_free (contents);

  if (prop->labels != NULL) {
    g_print ("Loaded image label file successfully. %u labels loaded.",
        prop->total_labels);
  }
  return;
}

/** @brief Handle "PROP_LABELS" for set-property */
static gint
_gtfc_setprop_LABELS (GstSscmaYolov5 * priv,
    GstSscmaYolov5Properties * prop, const GValue * value)
{
  gint status = 0;
  const gchar *model_labels = g_value_get_string (value);
  GstSscmaYolov5Properties _prop;

  if (!model_labels) {
    g_print ("Invalid model provided to the tensor-filter.");
    return 0;
  }
  loadImageLabels (model_labels, prop);
  return 0;
}

/** @brief Handle "PROP_INPUT" for set-property */
static gint
_gtfc_setprop_DIMENSION (GstSscmaYolov5 * priv,
    const GValue * value, const gboolean is_input)
{
  GstSscmaYolov5Properties *prop;
  GstTensorsInfo *info;
  unsigned int *rank;
  int configured;

  prop = &priv->prop;
  if(is_input){
    info = &prop->input_meta;
    rank = prop->input_ranks;
    configured = prop->input_configured;
  }else{
    info = &prop->output_meta;
    rank = prop->output_ranks;
    configured = prop->output_configured;
  }

  if (!configured && value) {
    guint num_dims;
    gchar **str_dims;
    guint i;

    str_dims = g_strsplit_set (g_value_get_string (value), ",.", -1);
    num_dims = g_strv_length (str_dims);

    if (num_dims > NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT) {
      g_print ("Invalid param, dimensions (%d) max (%d)\n",
          num_dims, NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT);

      num_dims = NNS_TENSOR_SIZE_LIMIT + NNS_TENSOR_SIZE_EXTRA_LIMIT;
    }

    for (i = 0; i < num_dims; ++i) {
      rank[i] = gst_tensor_parse_dimension (str_dims[i],
          gst_tensors_info_get_nth_info (info, i)->dimension);
    }
    g_strfreev (str_dims);

    if (num_dims > 0) {
      if (info->num_tensors > 0 && info->num_tensors != num_dims) {
        g_print
            ("Invalid dimension, given param does not match with old value.");
      }

      info->num_tensors = num_dims;
    }
  } else if (value) {
    /** Once configured, it cannot be changed in runtime for now */
    g_print
        ("Cannot change dimension once the element/pipeline is configured.");
  }
  return 0;
}

/** @brief Handle "PROP_INPUTTYPE" and "PROP_OUTPUTTYPE" for set-property */
static gint
_gtfc_setprop_TYPE (GstSscmaYolov5 * priv,
    const GValue * value, const gboolean is_input)
{
  GstSscmaYolov5Properties *prop;
  GstTensorsInfo *info;
  int configured;

  prop = &priv->prop;

  if (is_input) {
    info = &prop->input_meta;
    configured = prop->input_configured;
  } else {
    info = &prop->output_meta;
    configured = prop->output_configured;
  }

  if (!configured && value) {
    guint num_types;

    num_types = gst_tensors_info_parse_types_string (info,
        g_value_get_string (value));

    if (num_types > 0) {
      if (info->num_tensors > 0 && info->num_tensors != num_types) {
        g_print ("Invalid type, given param does not match with old value.");
      }

      info->num_tensors = num_types;
    }
  } else if (value) {
    /** Once configured, it cannot be changed in runtime for now */
    g_print ("Cannot change type once the element/pipeline is configured.");
  }
  return 0;
}

static void
gst_sscma_yolov5_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstSscmaYolov5 *self = GST_SWIFT_YOLOV5 (object);
  GstSscmaYolov5Properties *prop;
  gint status = 0;
  // UNUSED (pspec);

  prop = &self->prop;
  switch (prop_id) {
    // 输入模型 mode=xxx,xxx（可为多个）
    case PROP_MODEL:
      status = _gtfc_setprop_MODEL (self, prop, value);
      break;
    // 标签文件配置 labels=xxx
    case PROP_MODE_LABELS:
      status = _gtfc_setprop_LABELS (self, prop, value);
      break;
    // 输入视频大小 input=320:320:3
    case PROP_INPUT:
      status = _gtfc_setprop_DIMENSION (self, value, TRUE);
      break;
    // 模型输出大小 output=85:6300:1:1 注:第2个维度为识别结果个数 后面画框会用到
    case PROP_OUTPUT:
      status = _gtfc_setprop_DIMENSION (self, value, FALSE);
      break;
    // TODO:输入格式 inputformat=RGB
    case PROP_INPUTFORMAT:
      // status = _gtfc_setprop_FORMAT (self, value);
      break;
    // 输出类型 inputtype=float32
    case PROP_INPUTTYPE:
      status = _gtfc_setprop_TYPE (self, value, TRUE);
      break;
    // 输出类型 outputtype=float32
    case PROP_OUTPUTTYPE:
      status = _gtfc_setprop_TYPE (self, value, FALSE);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
  if(status != 0){
    g_print ("Invalid param, please check your param.\n");
  }
}

static void
gst_sscma_yolov5_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstSscmaYolov5 *filter = GST_SWIFT_YOLOV5 (object);

  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * @brief This function handles sink event.
 */
static gboolean
gst_sscma_yolov5_sink_event (GstPad * pad, GstObject * parent,
    GstEvent * event)
{
  GstSscmaYolov5 *self = GST_SWIFT_YOLOV5 (parent);
  gboolean ret;
  GST_DEBUG_OBJECT (self, "Received %s event: %" GST_PTR_FORMAT,
      GST_EVENT_TYPE_NAME (event), event);

  switch (GST_EVENT_TYPE (event)) {
    case GST_EVENT_STREAM_START:
    {
      // load model
      if (self->prop.num_models > 0) {
        net.load_param(self->prop.model_files[1]);
        net.load_model(self->prop.model_files[0]);
      }
      ret = gst_pad_event_default (pad, parent, event);
      break;
    }
    case GST_EVENT_CAPS:
    {
      GstCaps *in_caps;
      gst_event_parse_caps (event, &in_caps);

      if (gst_sscma_yolov5_parse_caps (self, in_caps)) {
        // ret = gst_sscma_yolov5_update_caps (self);
        gst_event_unref (event);
      } else {
        gst_event_unref (event);
        // ret = FALSE;
      }
      ret = gst_pad_event_default (pad, parent, event);
      break;
    }
    default:
      break;
  }

  return ret;
}

/**
 * @brief This function handles sink pad query.
 */
static gboolean
gst_sscma_yolov5_sink_query (GstPad * pad, GstObject * parent,
    GstQuery * query)
{
  GstSscmaYolov5 *self = GST_SWIFT_YOLOV5 (parent);
  gboolean ret;
  GST_DEBUG_OBJECT (self, "Received %s query: %" GST_PTR_FORMAT,
      GST_QUERY_TYPE_NAME (query), query);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CAPS:
    {
      GstCaps *caps;
      GstCaps *filter;

      gst_query_parse_caps (query, &filter);
      // TODO:输入应固定为模型需要的视频大小
      // caps = gst_caps_new_any ();
      // gst_query_set_caps_result (query, caps);
      // gst_caps_unref (caps);
      ret = gst_pad_query_default (pad, parent, query);

    }
    case GST_QUERY_ACCEPT_CAPS:
    {
      GstCaps *caps;
      GstCaps *template_caps;
      gboolean res = FALSE;
      // TODO:根据下一个插件选择输出类型
      // gst_query_parse_accept_caps (query, &caps);

      // if (gst_caps_is_fixed (caps)) {
      //   template_caps = gst_pad_get_pad_template_caps (pad);

      //   res = gst_caps_can_intersect (template_caps, caps);
      //   gst_caps_unref (template_caps);
      // }

      // gst_query_set_accept_caps_result (query, res);
      ret = gst_pad_query_default (pad, parent, query);

    }
    default:
      ret = gst_pad_query_default (pad, parent, query);
      break;
  }

  return ret;
}

/**
 * @brief This function handles src pad query.
 */
static gboolean
gst_sscma_yolov5_src_query (GstPad * pad, GstObject * parent,
    GstQuery * query)
{
  GstSscmaYolov5 *self = GST_SWIFT_YOLOV5 (parent);
  gboolean ret;
  GST_DEBUG_OBJECT (self, "Received %s query: %" GST_PTR_FORMAT,
      GST_QUERY_TYPE_NAME (query), query);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_CAPS:
    {
      GstCaps *caps;
      GstCaps *filter;

      gst_query_parse_caps (query, &filter);

      caps = gst_caps_new_simple("video/x-raw",
                                        "width", G_TYPE_INT, 320,
                                        "height", G_TYPE_INT, 320,
                                        "format", G_TYPE_STRING, "RGB",
                                        NULL);
      gst_query_set_caps_result (query, caps);
      gst_caps_unref (caps);
      ret = TRUE;
      ret = gst_pad_query_default (pad, parent, query);
      break;
    }
    default:
      ret = gst_pad_query_default (pad, parent, query);
      break;
  }

  return ret;
}
/**
 * @brief Check input paramters for gst_tensor_filter_transform ();
 */
static GstFlowReturn
gst_swift_yolov5_validate (GstSscmaYolov5Properties * prop,
    GstBuffer * inbuf)
{

  return GST_FLOW_OK;
}

/**
 * @brief Chain function, this function does the actual processing.
 */
static GstFlowReturn
gst_sscma_yolov5_chain (GstPad * pad, GstObject * parent, GstBuffer * buf)
{
  GstSscmaYolov5 *self = GST_SWIFT_YOLOV5 (parent);
  GstSscmaYolov5Properties *prop = &self->prop;
  GstBuffer *inbuf, *outbuf;
  GstMapInfo src_info, dest_info;
  GstTensorsInfo *info;
  GstTensorInfo *_info;
  gsize buf_size, frame_size, out_size, type;
  guint color, width, height, max_index, cIdx_max;
  gfloat *data, max_index_val;
  GArray *results = NULL;
  // UNUSED (pad);

  ncnn::Mat in_pad;
  ncnn::Mat out;
  ncnn::Extractor ex = net.create_extractor();

  /* 0. validate input */
  buf_size = gst_buffer_get_size (buf);
  g_return_val_if_fail (buf_size > 0, GST_FLOW_ERROR);

  /* 1. Check all properties. */
  GstFlowReturn retval = gst_swift_yolov5_validate (prop, buf);
  if (retval != GST_FLOW_OK)
    return retval;

  /* 2. preprocess data */
  // g_assert (self->tensors_configured);
  info = &prop->input_meta;
  color = info->info[0].dimension[0];
  width = info->info[0].dimension[1];
  height = info->info[0].dimension[2];
  type = tensor_element_size[info->info[0].type];
  /** type * colorspace * width * height */
  frame_size = type * color * width * height;

  /** supposed 1 frame in buffer */
  g_assert ((buf_size / frame_size) == 1);

  if (!gst_buffer_map (buf, &src_info, GST_MAP_READ)) {
    g_print
        ("tensor_converter: Cannot map src buffer at tensor_converter/video. The incoming buffer (GstBuffer) for the sinkpad of tensor_converter cannot be mapped for reading.\n");
    goto error;
  }
  /* output size*/
  out_size = tensor_element_size[prop->output_meta.info[0].type];
  for(int i = 0; i <3; i++){
    out_size *= prop->output_meta.info[0].dimension[i];
  }

  inbuf = gst_buffer_new_and_alloc (out_size);
  gst_buffer_memset (inbuf, 0, 0, out_size);
  if (!gst_buffer_map (inbuf, &dest_info, GST_MAP_WRITE)) {
    g_print
        ("tensor_converter: Cannot map dest buffer at tensor_converter/video. The outgoing buffer (GstBuffer) for the srcpad of tensor_converter cannot be mapped for writing.\n");
    gst_buffer_unmap (buf, &src_info);
    goto error;
  }

  /* 3. inference*/
  for (uint i = 0; i < prop->input_meta.num_tensors; ++i) {
    std::vector<int64_t> input_shape;

    _info = gst_tensors_info_get_nth_info (info, i);

    input_shape.assign (&_info->dimension[0], &_info->dimension[0] + NNS_TENSOR_RANK_LIMIT);
    input_shape.resize (prop->input_ranks[i]);
    std::reverse (input_shape.begin (), input_shape.end ());

    in_pad = ncnn::Mat::from_pixels_resize(src_info.data, ncnn::Mat::PIXEL_RGB, input_shape[1], input_shape[2],input_shape[1],input_shape[2]);
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);
    ex.input("in0", in_pad);
    ex.extract("out0", out);
    g_assert (out.total() * out.elemsize == out_size);
    memcpy (dest_info.data, out.data, out_size);
  }

  /* 4. Post-processing of the data*/
  cIdx_max = prop->total_labels + DETECTION_NUM_INFO;
  results = g_array_sized_new (FALSE, TRUE, sizeof (detectedObject), prop->output_meta.info[0].dimension[2]);
  data = (float *)dest_info.data;
  for (int delect_num = 0; delect_num < prop->output_meta.info[0].dimension[1]; delect_num++) {
    max_index_val = 0;
    max_index = 0;
    // Find the class with the maximum confidence
    // todo: 可配置阈值
    for (int i = DETECTION_NUM_INFO; i < prop->total_labels; ++i) {
      if (data[delect_num * cIdx_max + i] > max_index_val) {
        max_index_val = data[delect_num * cIdx_max + i];
        max_index = i;
      }
    }

    // If the maximum confidence is greater than the threshold, then the result is valid
    // TODO ：可配置阈值
    if (max_index_val * data[delect_num * cIdx_max + 4] > 2500) {
      detectedObject object;
      float cx, cy, w, h;
      cx = data[delect_num * cIdx_max + 0];
      cy = data[delect_num * cIdx_max + 1];
      w = data[delect_num * cIdx_max + 2];
      h = data[delect_num * cIdx_max + 3];

      // todo:可配置是否归一化
      // if (!is_output_scaled) {
      //   cx *= (float) width;
      //   cy *= (float) height;
      //   w *= (float) width;
      //   h *= (float) height;
      // }

      object.x = (int) (MAX (0.f, (cx - w / 2.f)));
      object.y = (int) (MAX (0.f, (cy - h / 2.f)));
      object.width = (int) (MIN ((float) width, w));
      object.height = (int) (MIN ((float) height, h));

      object.prob = max_index_val * data[delect_num * cIdx_max + 4];
      object.class_id = max_index - DETECTION_NUM_INFO;
      object.tracking_id = int(max_index_val);
      object.valid = TRUE;
      g_array_append_val (results, object);
    }
  }
  /* clear inbuf */
  gst_buffer_unmap (inbuf, &dest_info);
  gst_buffer_unref (inbuf);
  // todo: 可配置阈值
  nms (results, 0.25);

  /* 5. draw box */
  outbuf = gst_buffer_new_and_alloc (frame_size);
  gst_buffer_memset (outbuf, 0, 0, frame_size);
  if (!gst_buffer_map (outbuf, &dest_info, GST_MAP_WRITE)) {
    g_print
        ("tensor_converter: Cannot map dest buffer at tensor_converter/video. The outgoing buffer (GstBuffer) for the srcpad of tensor_converter cannot be mapped for writing.\n");
    gst_buffer_unmap (buf, &src_info);
    goto error;
  }

  memcpy (dest_info.data, src_info.data, dest_info.size);
  gst_buffer_unmap (buf, &src_info);
  gst_buffer_unref (buf);
  // TODO：支持多个输出格式 主要是RGB RGBA
  draw (&dest_info, prop, results);
  g_array_free (results, TRUE);
  
  gst_buffer_unmap (outbuf, &dest_info);
  return gst_pad_push (self->srcpad, outbuf);
error:
  if (inbuf)
    gst_buffer_unref (inbuf);
  if (outbuf)
    gst_buffer_unref (outbuf);
  gst_buffer_unref (buf);
  return GST_FLOW_ERROR;
}

/**
 * @brief Set the tensors info structure from video info (internal static function)
 * @param self this pointer to GstSscmaYolov5
 * @param caps caps for media stream
 * @param info tensors info structure to be filled
 * @note Change dimension if tensor contains N frames.
 * @return TRUE if supported type
 */
static gboolean
gst_sscma_yolov5_parse_video (GstSscmaYolov5 * self,
    const GstCaps * caps, GstTensorsInfo * info)
{
  /**
   * Refer: https://www.tensorflow.org/api_docs/python/tf/summary/image
   * A 4-D uint8 or float32 Tensor of shape [batch_size, height, width, channels]
   * where channels is 1, 3, or 4.
   */
  GstVideoInfo vinfo;
  GstVideoFormat format;
  gint width, height, views;
  guint i;

  g_return_val_if_fail (info != NULL, FALSE);

  gst_tensors_info_init (info);

  gst_video_info_init (&vinfo);
  if (!gst_video_info_from_caps (&vinfo, caps)) {
    char *capstr = gst_caps_to_string (caps);
    GST_ERROR_OBJECT (self,
        "Failed to get video info from caps; gst_video_info_from_caps (&info, \"%s\") has returned FALSE, which means the given caps cannot be parsed as a video.",
        capstr);
    g_free (capstr);
    return FALSE;
  }

  format = GST_VIDEO_INFO_FORMAT (&vinfo);
  width = GST_VIDEO_INFO_WIDTH (&vinfo);
  height = GST_VIDEO_INFO_HEIGHT (&vinfo);
  views = GST_VIDEO_INFO_VIEWS (&vinfo);
  if (views > 1) {
    GST_WARNING_OBJECT (self,
        "Incoming video caps should have 'views=(int)1 but has views=(int)%d - ignoring all but view #0. \n",
        views);
  }

  info->num_tensors = 1;
  self->prop.input_meta.num_tensors = 1;
  /* [color-space][width][height][frames] */
  switch (format) {
    case GST_VIDEO_FORMAT_GRAY8:
      info->info[0].type = _TENOR_UINT8;
      info->info[0].dimension[0] = 1;
      break;
    case GST_VIDEO_FORMAT_GRAY16_BE:
    case GST_VIDEO_FORMAT_GRAY16_LE:
      info->info[0].type = _TENOR_UINT16;
      info->info[0].dimension[0] = 1;
      break;
    case GST_VIDEO_FORMAT_RGB:
    case GST_VIDEO_FORMAT_BGR:
      info->info[0].type = _TENOR_UINT8;
      info->info[0].dimension[0] = 3;
      break;
    case GST_VIDEO_FORMAT_RGBx:
    case GST_VIDEO_FORMAT_BGRx:
    case GST_VIDEO_FORMAT_xRGB:
    case GST_VIDEO_FORMAT_xBGR:
    case GST_VIDEO_FORMAT_RGBA:
    case GST_VIDEO_FORMAT_BGRA:
    case GST_VIDEO_FORMAT_ARGB:
    case GST_VIDEO_FORMAT_ABGR:
      info->info[0].type = _TENOR_UINT8;
      info->info[0].dimension[0] = 4;
      break;
    default:
      GST_WARNING_OBJECT (self,
          "The given video caps with format \"%s\" is not supported. Please use GRAY8, GRAY16_LE, GRAY16_BE, RGB, BGR, RGBx, BGRx, xRGB, xBGR, RGBA, BGRA, ARGB, or ABGR.\n",
          GST_STR_NULL (gst_video_format_to_string (format)));
      break;
  }

  info->info[0].dimension[1] = width;
  info->info[0].dimension[2] = height;

  /* Supposed 1 frame in tensor, change dimension[3] if tensor contains N frames. */
  info->info[0].dimension[3] = 1;
  for (i = 4; i < NNS_TENSOR_RANK_LIMIT; i++)
    info->info[0].dimension[i] = 0;

  self->rate_n = GST_VIDEO_INFO_FPS_N (&vinfo);
  self->rate_d = GST_VIDEO_INFO_FPS_D (&vinfo);

  /**
   * @todo The actual list is much longer, fill them.
   * (read https://gstreamer.freedesktop.org/documentation/design/mediatype-video-raw.html)
   */
  switch (format) {
    case GST_VIDEO_FORMAT_GRAY8:
    case GST_VIDEO_FORMAT_RGB:
    case GST_VIDEO_FORMAT_BGR:
    case GST_VIDEO_FORMAT_I420:
      if (width % 4) {
        GST_ERROR_OBJECT (self,
          "\nYOUR STREAM INFOURATION INCURS PERFORMANCE DETERIORATION!\n"
          "Please use 4 x n as image width for inputs; the width of your input is %d.\n",
          width);
        return NULL;
      }
      break;
    default:
      break;
  }

  return (info->info[0].type != _TENOR_END);
}

/**
 * @brief Get pad caps for caps negotiation.
 */
static GstCaps *
gst_sscma_yolov5_query_caps (GstSscmaYolov5 * self, GstPad * pad,
    GstCaps * filter)
{
  GstCaps *caps;

  caps = gst_pad_get_current_caps (pad);
  if (!caps) {
    caps = gst_pad_get_pad_template_caps (pad);
  }

  if (pad == self->sinkpad) {
    GstCaps *media_caps;

    // media_caps = gst_sscma_yolov5_get_possible_media_caps (self);
    if (media_caps) {
      /* intersect with pad caps */
      GstCaps *tmp = gst_caps_intersect_full (media_caps, caps,
          GST_CAPS_INTERSECT_FIRST);
      gst_caps_unref (caps);
      caps = tmp;

      gst_caps_unref (media_caps);
    }
  }

  if (filter) {
    GstCaps *intersection;
    intersection =
        gst_caps_intersect_full (filter, caps, GST_CAPS_INTERSECT_FIRST);

    gst_caps_unref (caps);
    caps = intersection;
  }
  return caps;
}

/**
 * @brief Parse caps and set tensors info.
 */
static gboolean
gst_sscma_yolov5_parse_caps (GstSscmaYolov5 * self,
    const GstCaps * caps)
{
  GstStructure *structure;
  GstTensorsInfo info;
  const gchar *name;

  gint frames_dim = -1; /** dimension index of frames in infoured tensors */

  g_return_val_if_fail (caps != NULL, FALSE);
  g_return_val_if_fail (gst_caps_is_fixed (caps), FALSE);

  structure = gst_caps_get_structure (caps, 0);

  name = gst_structure_get_name (structure);
  g_return_val_if_fail (name != NULL, FALSE);

  if (!g_str_has_prefix (name, "video/")) {
    GST_ERROR_OBJECT (self,
        "Failed to configure tensor: sink must be video stream.");
    return FALSE;
  }

  if (!gst_sscma_yolov5_parse_video (self, caps, &info)) {
    char *capstr = gst_caps_to_string (caps);
    GST_ERROR_OBJECT (self,
        "Failed to configure tensor from gst cap \"%s\" for video streams.",
        capstr);
    g_free (capstr);
    return FALSE;
  }

  frames_dim = 3;
  /** set the number of frames in dimension */
  info.info[0].dimension[frames_dim] = 1;
  if (!gst_tensors_info_validate (&info)) {
    GST_ERROR_OBJECT (self,
        "Failed, the given tensor info is invalid. Please check the given tensor info.");
    return FALSE;
  }
  // self->tensors_configured = TRUE;
  self->prop.input_meta = info;
  return TRUE;
}

/**
 * @brief Update src pad caps from tensors config.
 */
static gboolean
gst_sscma_yolov5_update_caps (GstSscmaYolov5 * self)
{
  GstTensorsInfo *info;
  GstCaps *curr_caps, *out_caps;
  gboolean ret = FALSE;

  info = &self->prop.input_meta;
  // out cap is ANY
  out_caps = gst_caps_new_any ();

  /* Update src pad caps if it is different. */
  curr_caps = gst_pad_get_current_caps (self->srcpad);
  if (curr_caps == NULL || !gst_caps_is_equal (curr_caps, out_caps)) {
    ret = gst_pad_set_caps (self->srcpad, out_caps);
  }

  if (curr_caps)
    gst_caps_unref (curr_caps);

  gst_caps_unref (out_caps);
  return ret;
}

/**
 * @brief Compare Function for g_array_sort with detectedObject.
 */
static gint
compare_detection (gconstpointer _a, gconstpointer _b)
{
  const detectedObject *a = _a;
  const detectedObject *b = _b;

  /* Larger comes first */
  return (a->prob > b->prob) ? -1 : ((a->prob == b->prob) ? 0 : 1);
}

/**
 * @brief Calculate the intersected surface
 */
static gfloat
iou (detectedObject * a, detectedObject * b)
{
  int x1 = MAX (a->x, b->x);
  int y1 = MAX (a->y, b->y);
  int x2 = MIN (a->x + a->width, b->x + b->width);
  int y2 = MIN (a->y + a->height, b->y + b->height);
  int w = MAX (0, (x2 - x1 + 1));
  int h = MAX (0, (y2 - y1 + 1));
  float inter = w * h;
  float areaA = a->width * a->height;
  float areaB = b->width * b->height;
  float o = inter / (areaA + areaB - inter);
  return (o >= 0) ? o : 0;
}

/**
 * @brief Apply NMS to the given results (objects[MOBILENET_SSD_DETECTION_MAX])
 * @param[in/out] results The results to be filtered with nms
 */
static void
nms (GArray * results, gfloat threshold)
{
  guint boxes_size;
  guint i, j;

  boxes_size = results->len;
  if (boxes_size == 0U)
    return;

  g_array_sort (results, compare_detection);

  for (i = 0; i < boxes_size; i++) {
    detectedObject *a = &g_array_index (results, detectedObject, i);
    if (a->valid == TRUE) {
      for (j = i + 1; j < boxes_size; j++) {
        detectedObject *b = &g_array_index (results, detectedObject, j);
        if (b->valid == TRUE) {
          if (iou (a, b) > threshold) {
            b->valid = FALSE;
          }
        }
      }
    }
  }

  i = 0;
  do {
    detectedObject *a = &g_array_index (results, detectedObject, i);
    if (a->valid == FALSE)
      g_array_remove_index (results, i);
    else
      i++;
  } while (i < results->len);

}

/**
 * @brief Draw with the given results (objects[MOBILENET_SSD_DETECTION_MAX]) to the output buffer
 * @param[out] out_info The output buffer (RGBA plain)
 * @param[in] prop The bounding-box internal data.
 * @param[in] results The final results to be drawn.
 */
static void
draw (GstMapInfo * out_info, GstSscmaYolov5Properties *prop, GArray * results)
{
  uint8_t *frame = (uint8_t *) out_info->data;        /* Let's draw per pixel (4bytes) */
  unsigned int i;
  guint color = prop->input_meta.info[0].dimension[0];
  guint width = prop->input_meta.info[0].dimension[1];
  guint height = prop->input_meta.info[0].dimension[2];
  for (i = 0; i < results->len; i++) {
    int x1, x2, y1, y2;         /* Box positions on the output surface */
    int j;
    uint8_t *pos1, *pos2;
    detectedObject *a = &g_array_index (results, detectedObject, i);

    if ((a->class_id < 0 ||
                a->class_id >= (int) prop->total_labels)) {
      /** @todo make it "logw_once" after we get logw_once API. */
      g_print ("Invalid class found with tensordec-boundingbox.c.\n");
      continue;
    }

    /* 1. Draw Boxes */
    x1 =  a->x;
    x2 = MIN (width - 1, (a->x + a->width));
    y1 = a->y;
    y2 = MIN (height - 1,(a->y + a->height));
    /* 1-1. Horizontal */
    pos1 = &frame[(y1 * width + x1) * 3];
    pos2 = &frame[(y2 * width + x1) * 3];
    for (j = x1; j <= x2; j++) {
      *pos1 = PIXEL_VALUE;
      *pos2 = PIXEL_VALUE;
      pos1 += 3;
      pos2 += 3;
    }

    /* 1-2. Vertical */
    pos1 = &frame[((y1 + 1) * width + x1) * 3];
    pos2 = &frame[((y1 + 1) * width + x2) * 3];
    for (j = y1 + 1; j < y2; j++) {
      *pos1 = PIXEL_VALUE;
      *pos2 = PIXEL_VALUE;
      pos1 += width * 3;
      pos2 += width * 3;
    }

    /* 2. Write Labels + tracking ID */
    g_autofree gchar *label = NULL;
    gsize label_len;
    label = g_strdup_printf ("%s %d", prop->labels[a->class_id],
            a->tracking_id);
    label_len = strlen (label);
    /* x1 is the same: x1 = MAX (0, (width * a->x) / bdata->i_width); */
    y1 = MAX (0, (y1 - 14));
    pos1 = &frame[(y1 * width + x1) * 3];
    for (guint j = 0; j < label_len; j++) {
      unsigned int char_index = label[j];
      if (char_index < 32 || char_index >= 127) {
        /* It's not ASCII */
        char_index = '*';
      }
      char_index -= 32;
      if ((x1 + 8) > (int) width)
        break;                /* Stop drawing if it may overfill */
      pos2 = pos1;
      for (y2 = 0; y2 < 13; y2++) {
        /* 13 : character height */
        for (x2 = 0; x2 < 8; x2++) {
          /* 8: character width */
          *(pos2 + x2 * 3) = rasters[char_index][13-y2] & (1 << (7 - x2)) ?
              PIXEL_VALUE : 0;
        }
        pos2 += width * 3;
      }
      x1 += 9;
      pos1 += 9 * 3;              /* charater width + 1px */
    }
  }
}
/* entry point to initialize the plug-in
 * initialize the plug-in itself
 * register the element factories and other features
 */
static gboolean
sscma_yolov5_init (GstPlugin * sscmayolov5)
{
  /* debug category for filtering log messages
   *
   * exchange the string 'Template sscmayolov5' with your description
   */
  GST_DEBUG_CATEGORY_INIT (gst_sscma_yolov5_debug, "sscmayolov5",
      0, "Template sscma yolov5");

  return GST_ELEMENT_REGISTER (sscma_yolov5, sscmayolov5);
}

/* PACKAGE: this is usually set by meson depending on some _INIT macro
 * in meson.build and then written into and defined in config.h, but we can
 * just set it ourselves here in case someone doesn't use meson to
 * compile this code. GST_PLUGIN_DEFINE needs PACKAGE to be defined.
 */
#ifndef PACKAGE
#define PACKAGE "sscmayolov5"
#endif

/* gstreamer looks for this structure to register sscmayolov5s
 *
 * exchange the string 'Template sscmayolov5' with your sscmayolov5 description
 */
GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    sscmayolov5,
    "sscma_yolov5",
    sscma_yolov5_init,
    PACKAGE_VERSION, GST_LICENSE, GST_PACKAGE_NAME, GST_PACKAGE_ORIGIN)
