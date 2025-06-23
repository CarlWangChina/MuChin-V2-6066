import sys
sys_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorboardX/proto/api.proto',
  package='tensorboardX.hparam',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x1ctensorboardX/proto/api.proto\x12\x13tensorboardX.hparam\x1a\x1cgoogle/protobuf/struct.proto\"\xc6\x01\n\nExperiment\x12\x0c\n\x04name\x18\x06 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x01 \x01(\t\x12\x0c\n\x04user\x18\x02 \x01(\t\x12\x19\n\x11time_created_secs\x18\x03 \x01(\x01\x12\x35\n\x0chparam_infos\x18\x04 \x03(\x0b\x32\x1f.tensorboardX.hparam.HParamInfo\x12\x35\n\x0cmetric_infos\x18\x05 \x03(\x0b\x32\x1f.tensorboardX.hparam.MetricInfo\"\xed\x01\n\nHParamInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x14\n\x0c\x64isplay_name\x18\x02 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x03 \x01(\t\x12+\n\x04type\x18\x04 \x01(\x0e\x32\x1d.tensorboardX.hparam.DataType\x12\x35\n\x0f\x64omain_discrete\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.ListValueH\x00\x12\x38\n\x0f\x64omain_interval\x18\x06 \x01(\x0b\x32\x1d.tensorboardX.hparam.IntervalH\x00\x42\x08\n\x06\x64omain\"0\n\x08Interval\x12\x11\n\tmin_value\x18\x01 \x01(\x01\x12\x11\n\tmax_value\x18\x02 \x01(\x01\"(\n\nMetricName\x12\r\n\x05group\x18\x01 \x01(\t\x12\x0b\n\x03tag\x18\x02 \x01(\t\"\x9e\x01\n\nMetricInfo\x12-\n\x04name\x18\x01 \x01(\x0b\x32\x1f.tensorboardX.hparam.MetricName\x12\x14\n\x0c\x64isplay_name\x18\x03 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x04 \x01(\t\x12\x36\n\x0c\x64\x61taset_type\x18\x05 \x01(\x0e\x32 .tensorboardX.hparam.DatasetType\"\xa3\x02\n\x0cSessionGroup\x12\x0c\n\x04name\x18\x01 \x01(\t\x12?\n\x07hparams\x18\x02 \x03(\x0b\x32..tensorboardX.hparam.SessionGroup.HparamsEntry\x12\x37\n\rmetric_values\x18\x03 \x03(\x0b\x32 .tensorboardX.hparam.MetricValue\x12.\n\x08sessions\x18\x04 \x03(\x0b\x32\x1c.tensorboardX.hparam.Session\x12\x13\n\x0bmonitor_url\x18\x05 \x01(\t\x1a\x46\n\x0cHparamsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12%\n\x05value\x18\x02 \x01(\x0b\x32\x16.google.protobuf.Value:\x02\x38\x01\"z\n\x0bMetricValue\x12-\n\x04name\x18\x01 \x01(\x0b\x32\x1f.tensorboardX.hparam.MetricName\x12\r\n\x05value\x18\x02 \x01(\x01\x12\x15\n\rtraining_step\x18\x03 \x01(\x05\x12\x16\n\x0ewall_time_secs\x18\x04 \x01(\x01\"\xd5\x01\n\x07Session\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x17\n\x0fstart_time_secs\x18\x02 \x01(\x01\x12\x15\n\rend_time_secs\x18\x03 \x01(\x01\x12+\n\x06status\x18\x04 \x01(\x0e\x32\x1b.tensorboardX.hparam.Status\x12\x11\n\tmodel_uri\x18\x05 \x01(\t\x12\x37\n\rmetric_values\x18\x06 \x03(\x0b\x32 .tensorboardX.hparam.MetricValue\x12\x13\n\x0bmonitor_url\x18\x07 \x01(\t\"/\n\x14GetExperimentRequest\x12\x17\n\x0f\x65xperiment_name\x18\x01 \x01(\t\"\xc4\x02\n\x18ListSessionGroupsRequest\x12\x17\n\x0f\x65xperiment_name\x18\x06 \x01(\t\x12\x35\n\x10\x61llowed_statuses\x18\x07 \x03(\x0e\x32\x1b.tensorboardX.hparam.Status\x12\x32\n\ncol_params\x18\x01 \x03(\x0b\x32\x1e.tensorboardX.hparam.ColParams\x12>\n\x10\x61ggregation_type\x18\x02 \x01(\x0e\x32$.tensorboardX.hparam.AggregationType\x12;\n\x12\x61ggregation_metric\x18\x03 \x01(\x0b\x32\x1f.tensorboardX.hparam.MetricName\x12\x13\n\x0bstart_index\x18\x04 \x01(\x05\x12\x12\n\nslice_size\x18\x05 \x01(\x05\"\xd9\x02\n\tColParams\x12\x31\n\x06metric\x18\x01 \x01(\x0b\x32\x1f.tensorboardX.hparam.MetricNameH\x00\x12\x10\n\x06hparam\x18\x02 \x01(\tH\x00\x12-\n\x05order\x18\x03 \x01(\x0e\x32\x1e.tensorboardX.hparam.SortOrder\x12\x1c\n\x14missing_values_first\x18\x04 \x01(\x08\x12\x17\n\rfilter_regexp\x18\x05 \x01(\tH\x01\x12\x38\n\x0f\x66ilter_interval\x18\x06 \x01(\x0b\x32\x1d.tensorboardX.hparam.IntervalH\x01\x12\x35\n\x0f\x66ilter_discrete\x18\x07 \x01(\x0b\x32\x1a.google.protobuf.ListValueH\x01\x12\x1e\n\x16\x65xclude_missing_values\x18\x08 \x01(\x08\x42\x06\n\x04nameB\x08\n\x06\x66ilter\"j\n\x19ListSessionGroupsResponse\x12\x39\n\x0esession_groups\x18\x01 \x03(\x0b\x32!.tensorboardX.hparam.SessionGroup\x12\x12\n\ntotal_size\x18\x03 \x01(\x05\"}\n\x16ListMetricEvalsRequest\x12\x17\n\x0f\x65xperiment_name\x18\x03 \x01(\t\x12\x14\n\x0csession_name\x18\x01 \x01(\t\x12\x34\n\x0bmetric_name\x18\x02 \x01(\x0b\x32\x1f.tensorboardX.hparam.MetricName*`\n\x08\x44\x61taType\x12\x13\n\x0f\x44\x41TA_TYPE_UNSET\x10\x00\x12\x14\n\x10\x44\x41TA_TYPE_STRING\x10\x01\x12\x12\n\x0e\x44\x41TA_TYPE_BOOL\x10\x02\x12\x15\n\x11\x44\x41TA_TYPE_FLOAT64\x10\x03*P\n\x0b\x44\x61tasetType\x12\x13\n\x0f\x44\x41TASET_UNKNOWN\x10\x00\x12\x14\n\x10\x44\x41TASET_TRAINING\x10\x01\x12\x16\n\x12\x44\x41TASET_VALIDATION\x10\x02*X\n\x06Status\x12\x12\n\x0eSTATUS_UNKNOWN\x10\x00\x12\x12\n\x0eSTATUS_SUCCESS\x10\x01\x12\x12\n\x0eSTATUS_FAILURE\x10\x02\x12\x12\n\x0eSTATUS_RUNNING\x10\x03*A\n\tSortOrder\x12\x15\n\x11ORDER_UNSPECIFIED\x10\x00\x12\r\n\tORDER_ASC\x10\x01\x12\x0e\n\nORDER_DESC\x10\x02*\x7f\n\x0f\x41ggregationType\x12\x15\n\x11\x41GGREGATION_UNSET\x10\x00\x12\x13\n\x0f\x41GGREGATION_AVG\x10\x01\x12\x16\n\x12\x41GGREGATION_MEDIAN\x10\x02\x12\x13\n\x0f\x41GGREGATION_MIN\x10\x03\x12\x13\n\x0f\x41GGREGATION_MAX\x10\x04\x62\x06proto3')
  ,
  dependencies=[google_dot_protobuf_dot_struct__pb2.DESCRIPTOR,])

_DATATYPE = _descriptor.EnumDescriptor(
  name='DataType',
  full_name='tensorboardX.hparam.DataType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DATA_TYPE_UNSET', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DATA_TYPE_STRING', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DATA_TYPE_BOOL', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DATA_TYPE_FLOAT64', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=2370,
  serialized_end=2466,
)
_sym_db.RegisterEnumDescriptor(_DATATYPE)
DataType = enum_type_wrapper.EnumTypeWrapper(_DATATYPE)

_DATASETTYPE = _descriptor.EnumDescriptor(
  name='DatasetType',
  full_name='tensorboardX.hparam.DatasetType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DATASET_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DATASET_TRAINING', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DATASET_VALIDATION', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=2468,
  serialized_end=2548,
)
_sym_db.RegisterEnumDescriptor(_DATASETTYPE)
DatasetType = enum_type_wrapper.EnumTypeWrapper(_DATASETTYPE)

_STATUS = _descriptor.EnumDescriptor(
  name='Status',
  full_name='tensorboardX.hparam.Status',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='STATUS_UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STATUS_SUCCESS', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STATUS_FAILURE', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STATUS_RUNNING', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=2550,
  serialized_end=2638,
)
_sym_db.RegisterEnumDescriptor(_STATUS)
Status = enum_type_wrapper.EnumTypeWrapper(_STATUS)

_SORTORDER = _descriptor.EnumDescriptor(
  name='SortOrder',
  full_name='tensorboardX.hparam.SortOrder',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='ORDER_UNSPECIFIED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ORDER_ASC', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ORDER_DESC', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=2640,
  serialized_end=2705,
)
_sym_db.RegisterEnumDescriptor(_SORTORDER)
SortOrder = enum_type_wrapper.EnumTypeWrapper(_SORTORDER)

_AGGREGATIONTYPE = _descriptor.EnumDescriptor(
  name='AggregationType',
  full_name='tensorboardX.hparam.AggregationType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AGGREGATION_UNSET', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AGGREGATION_AVG', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AGGREGATION_MEDIAN', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AGGREGATION_MIN', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AGGREGATION_MAX', index=4, number=4,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=2707,
  serialized_end=2834,
)
_sym_db.RegisterEnumDescriptor(_AGGREGATIONTYPE)
AggregationType = enum_type_wrapper.EnumTypeWrapper(_AGGREGATIONTYPE)

DATA_TYPE_UNSET = 0
DATA_TYPE_STRING = 1
DATA_TYPE_BOOL = 2
DATA_TYPE_FLOAT64 = 3
DATASET_UNKNOWN = 0
DATASET_TRAINING = 1
DATASET_VALIDATION = 2
STATUS_UNKNOWN = 0
STATUS_SUCCESS = 1
STATUS_FAILURE = 2
STATUS_RUNNING = 3
ORDER_UNSPECIFIED = 0
ORDER_ASC = 1
ORDER_DESC = 2
AGGREGATION_UNSET = 0
AGGREGATION_AVG = 1
AGGREGATION_MEDIAN = 2
AGGREGATION_MIN = 3
AGGREGATION_MAX = 4


_EXPERIMENT = _descriptor.Descriptor(
  name='Experiment',
  full_name='tensorboardX.hparam.Experiment',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='tensorboardX.hparam.Experiment.name', index=0,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='description', full_name='tensorboardX.hparam.Experiment.description', index=1,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='user', full_name='tensorboardX.hparam.Experiment.user', index=2,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='time_created_secs', full_name='tensorboardX.hparam.Experiment.time_created_secs', index=3,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hparam_infos', full_name='tensorboardX.hparam.Experiment.hparam_infos', index=4,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='metric_infos', full_name='tensorboardX.hparam.Experiment.metric_infos', index=5,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=84,
  serialized_end=282,
)


_HPARAMINFO = _descriptor.Descriptor(
  name='HParamInfo',
  full_name='tensorboardX.hparam.HParamInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='tensorboardX.hparam.HParamInfo.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='display_name', full_name='tensorboardX.hparam.HParamInfo.display_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='description', full_name='tensorboardX.hparam.HParamInfo.description', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='tensorboardX.hparam.HParamInfo.type', index=3,
      number=4, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='domain_discrete', full_name='tensorboardX.hparam.HParamInfo.domain_discrete', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='domain_interval', full_name='tensorboardX.hparam.HParamInfo.domain_interval', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='domain', full_name='tensorboardX.hparam.HParamInfo.domain',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=285,
  serialized_end=522,
)


_INTERVAL = _descriptor.Descriptor(
  name='Interval',
  full_name='tensorboardX.hparam.Interval',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='min_value', full_name='tensorboardX.hparam.Interval.min_value', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='max_value', full_name='tensorboardX.hparam.Interval.max_value', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=524,
  serialized_end=572,
)


_METRICNAME = _descriptor.Descriptor(
  name='MetricName',
  full_name='tensorboardX.hparam.MetricName',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='group', full_name='tensorboardX.hparam.MetricName.group', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tag', full_name='tensorboardX.hparam.MetricName.tag', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=574,
  serialized_end=614,
)


_METRICINFO = _descriptor.Descriptor(
  name='MetricInfo',
  full_name='tensorboardX.hparam.MetricInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='tensorboardX.hparam.MetricInfo.name', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='display_name', full_name='tensorboardX.hparam.MetricInfo.display_name', index=1,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='description', full_name='tensorboardX.hparam.MetricInfo.description', index=2,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dataset_type', full_name='tensorboardX.hparam.MetricInfo.dataset_type', index=3,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=617,
  serialized_end=775,
)


_SESSIONGROUP_HPARAMSENTRY = _descriptor.Descriptor(
  name='HparamsEntry',
  full_name='tensorboardX.hparam.SessionGroup.HparamsEntry',