project('gst-sscma-yolov5', 'cpp', version : '1.0.0.1', license : 'LGPL')

sscmayolov5_install_dir = join_paths(get_option('libdir'), 'gstreamer-1.0')

cxx = meson.get_compiler('cpp')

gst_version = meson.project_version()

api_version = '1.0'

plugin_c_args = ['-DHAVE_CONFIG_H']

cdata = configuration_data()
cdata.set_quoted('PACKAGE_VERSION', gst_version)
cdata.set_quoted('PACKAGE', 'gst-template-plugin')
cdata.set_quoted('GST_LICENSE', 'LGPL')
cdata.set_quoted('GST_API_VERSION', api_version)
cdata.set_quoted('GST_PACKAGE_NAME', 'GStreamer template Plug-ins')
cdata.set_quoted('GST_PACKAGE_ORIGIN', 'https://gstreamer.freedesktop.org')
configure_file(output : 'config.h', configuration : cdata)

gst_dep = dependency('gstreamer-1.0', version : '>=1.19',
    required : true, fallback : ['gstreamer', 'gst_dep'])
gstbase_dep = dependency('gstreamer-base-1.0', version : '>=1.19',
  fallback : ['gstreamer', 'gst_base_dep'])
gst_video_dep = dependency('gstreamer-video-1.0')

json_glib_dep = dependency('json-glib-1.0')


# The sscmayolov5 Plugin
 gstsscmayolov5_sources = [
  'src/gstsscmayolov5.cc',
  'src/tensor_info.cc'
  ]

# The sscmayolov5 include directories
gstsscmayolov5_include_dirs = [
  include_directories('src')
]

library_dep=cxx.find_library('ncnn', dirs : join_paths(meson.source_root(),'components/ncnn/build-aarch64-linux-gnu/install/lib'))
library_include_dir = 'components/ncnn/build-aarch64-linux-gnu/install/include/ncnn'

gstsscmayolov5 = library('gstsscmayolov5',
  gstsscmayolov5_sources,
  include_directories : [gstsscmayolov5_include_dirs, library_include_dir],
  dependencies : [gst_dep, gstbase_dep, gst_video_dep, library_dep, json_glib_dep],
  install : true,
  install_dir : sscmayolov5_install_dir,
  c_args: ['-fpermissive',plugin_c_args],
  cpp_args: ['-fpermissive','-fopenmp',plugin_c_args]
)