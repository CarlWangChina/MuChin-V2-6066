import sys
_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
from tensorboardX.proto import tensor_pb2 as tensorboardX_dot_proto_dot_tensor__pb2
from tensorboardX.proto import tensor_shape_pb2 as tensorboardX_dot_proto_dot_tensor__shape__pb2
from tensorboardX.proto import types_pb2 as tensorboardX_dot_proto_dot_types__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorboardX/proto/attr_value.proto',
  package='tensorboardX',
  syntax='proto3',
  serialized_options=_b('\n\030org.tensorflow.frameworkB\017AttrValueProtosP\001\370\001\001'),
  serialized_pb=_b('\n tensorboardX/proto/attr_value.proto\n\n\030org.tensorflow.frameworkB\017AttrValueProtosP\001\370\001\001\n\n\n\031tensorboardX/proto/attr_value.proto\x12\x0ctensorboardX\x1a,tensorboardX/proto/tensor.proto\x1a.tensorboardX/proto/tensor_shape.proto\x1a%tensorboardX/proto/types.proto\"\xd7\x03\n\tAttrValue\x12\t\n\x01s\x18\x02 \x01(\x0cH\x00\x12\t\n\x01i\x18\x03 \x01(\x03H\x00\x12\t\n\x01f\x18\x04 \x01(\x02H\x00\x12\t\n\x01b\x18\x05 \x01(\x08H\x00\x12,\n\x04type\x18\x06 \x01(\x0e\x32\x1e.tensorboardX.DataTypeH\x00\x12/\n\x05shape\x18\x07 \x01(\x0b\x32\x1e.tensorboardX.TensorShapeProtoH\x00\x12)\n\x06tensor\x18\x08 \x01(\x0b\x32\x17.tensorboardX.TensorProtoH\x00\x12.\n\x04list\x18\x01 \x01(\x0b\x32\x1e.tensorboardX.AttrValue.ListValueH\x00\x12.\n\x04func\x18\n \x01(\x0b\x32\x1e.tensorboardX.NameAttrListH\x00\x12\x14\n\x0bplaceholder\x18\t \x01(\tH\x00\x1a\x9c\x02\n\x08ListValue\x12\t\n\x01s\x18\x02 \x03(\x0c\x12\x11\n\x01i\x18\x03 \x03(\x03\x42\x02\x10\x01\x12\x11\n\x01f\x18\x04 \x03(\x02\x42\x02\x10\x01\x12\x11\n\x01b\x18\x05 \x03(\x08\x42\x02\x10\x01\x12,\n\x04type\x18\x06 \x03(\x0e\x32\x1e.tensorboardX.DataTypeB\x02\x10\x01\x12/\n\x05shape\x18\x07 \x03(\x0b\x32\x1e.tensorboardX.TensorShapeProto\x12)\n\x06tensor\x18\x08 \x03(\x0b\x32\x17.tensorboardX.TensorProto\x12.\n\x04func\x18\t \x03(\x0b\x32\x1e.tensorboardX.NameAttrListB\x07\n\x05value\"\x8a\x02\n\x0cNameAttrList\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x42\n\x04attr\x18\x02 \x03(\x0b\x32(.tensorboardX.NameAttrList.AttrEntryR\x04attr\x1a\\\n\tAttrEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x3d\n\x05value\x18\x02 \x01(\x0b\x32\x17.tensorboardX.AttrValueR\x05value:\x02\x38\x01\x42\x07\n\x05value')
)


_ATTRVALUE_LISTVALUE = _descriptor.Descriptor(
  name='ListValue',
  full_name='tensorboardX.AttrValue.ListValue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='s', full_name='tensorboardX.AttrValue.ListValue.s', index=0,
      number=2, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='i', full_name='tensorboardX.AttrValue.ListValue.i', index=1,
      number=3, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='f', full_name='tensorboardX.AttrValue.ListValue.f', index=2,
      number=4, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='b', full_name='tensorboardX.AttrValue.ListValue.b', index=3,
      number=5, type=8, cpp_type=7, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='tensorboardX.AttrValue.ListValue.type', index=4,
      number=6, type=14, cpp_type=8, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=_b('\020\001'), file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='tensorboardX.AttrValue.ListValue.shape', index=5,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tensor', full_name='tensorboardX.AttrValue.ListValue.tensor', index=6,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='func', full_name='tensorboardX.AttrValue.ListValue.func', index=7,
      number=9, type=11, cpp_type=10, label=3,
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
  serialized_start=476,
  serialized_end=717,
)

_ATTRVALUE = _descriptor.Descriptor(
  name='AttrValue',
  full_name='tensorboardX.AttrValue',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='s', full_name='tensorboardX.AttrValue.s', index=0,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='i', full_name='tensorboardX.AttrValue.i', index=1,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='f', full_name='tensorboardX.AttrValue.f', index=2,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='b', full_name='tensorboardX.AttrValue.b', index=3,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='type', full_name='tensorboardX.AttrValue.type', index=4,
      number=6, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='tensorboardX.AttrValue.shape', index=5,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tensor', full_name='tensorboardX.AttrValue.tensor', index=6,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='list', full_name='tensorboardX.AttrValue.list', index=7,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='func', full_name='tensorboardX.AttrValue.func', index=8,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='placeholder', full_name='tensorboardX.AttrValue.placeholder', index=9,
      number=9, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_ATTRVALUE_LISTVALUE, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='value', full_name='tensorboardX.AttrValue.value',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=158,
  serialized_end=726,
)

_NAMEATTRLIST_ATTRENTRY = _descriptor.Descriptor(
  name='AttrEntry',
  full_name='tensorboardX.NameAttrList.AttrEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorboardX.NameAttrList.AttrEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorboardX.NameAttrList.AttrEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
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
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=811,
  serialized_end=879,
)

_NAMEATTRLIST = _descriptor.Descriptor(
  name='NameAttrList',
  full_name='tensorboardX.NameAttrList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='tensorboardX.NameAttrList.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='attr', full_name='tensorboardX.NameAttrList.attr', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_NAMEATTRLIST_ATTRENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=729,
  serialized_end=879,
)

_ATTRVALUE_LISTVALUE.fields_by_name['type'].enum_type = tensorboardX_dot_proto_dot_types__pb2._DATATYPE
_ATTRVALUE_LISTVALUE.fields_by_name['shape'].message_type = tensorboardX_dot_proto_dot_tensor__shape__pb2._TENSORSHAPEPROTO
_ATTRVALUE_LISTVALUE.fields_by_name['tensor'].message_type = tensorboardX_dot_proto_dot_tensor__pb2._TENSORPROTO
_ATTRVALUE_LISTVALUE.fields_by_name['func'].message_type = _NAMEATTRLIST
_ATTRVALUE_LISTVALUE.containing_type = _ATTRVALUE
_ATTRVALUE.fields_by_name['type'].enum_type = tensorboardX_dot_proto_dot_types__pb2._DATATYPE
_ATTRVALUE.fields_by_name['shape'].message_type = tensorboardX_dot_proto_dot_tensor__shape__pb2._TENSORSHAPEPROTO
_ATTRVALUE.fields_by_name['tensor'].message_type = tensorboardX_dot_proto_dot_tensor__pb2._TENSORPROTO
_ATTRVALUE.fields_by_name['list'].message_type = _ATTRVALUE_LISTVALUE
_ATTRVALUE.fields_by_name['func'].message_type = _NAMEATTRLIST
_ATTRVALUE.oneofs_by_name['value'].fields.append(
  _ATTRVALUE.fields_by_name['s'])
_ATTRVALUE.fields_by_name['s'].containing_oneof = _ATTRVALUE.oneofs_by_name['value']
_ATTRVALUE.oneofs_by_name['value'].fields.append(
  _ATTRVALUE.fields_by_name['i'])
_ATTRVALUE.fields_by_name['i'].containing_oneof = _ATTRVALUE.oneofs_by_name['value']
_ATTRVALUE.oneofs_by_name['value'].fields.append(
  _ATTRVALUE.fields_by_name['f'])
_ATTRVALUE.fields_by_name['f'].containing_oneof = _ATTRVALUE.oneofs_by_name['value']
_ATTRVALUE.oneofs_by_name['value'].fields.append(
  _ATTRVALUE.fields_by_name['b'])
_ATTRVALUE.fields_by_name['b'].containing_oneof = _ATTRVALUE.oneofs_by_name['value']
_ATTRVALUE.oneofs_by_name['value'].fields.append(
  _ATTRVALUE.fields_by_name['type'])
_ATTRVALUE.fields_by_name['type'].containing_oneof = _ATTRVALUE.oneofs_by_name['value']
_ATTRVALUE.oneofs_by_name['value'].fields.append(
  _ATTRVALUE.fields_by_name['shape'])
_ATTRVALUE.fields_by_name['shape'].containing_oneof = _ATTRVALUE.oneofs_by_name['value']
_ATTRVALUE.oneofs_by_name['value'].fields.append(
  _ATTRVALUE.fields_by_name['tensor'])
_ATTRVALUE.fields_by_name['tensor'].containing_oneof = _ATTRVALUE.oneofs_by_name['value']
_ATTRVALUE.oneofs_by_name['value'].fields.append(
  _ATTRVALUE.fields_by_name['list'])
_ATTRVALUE.fields_by_name['list'].containing_oneof = _ATTRVALUE.oneofs_by_name['value']
_ATTRVALUE.oneofs_by_name['value'].fields.append(
  _ATTRVALUE.fields_by_name['func'])
_ATTRVALUE.fields_by_name['func'].containing_oneof = _ATTRVALUE.oneofs_by_name['value']
_ATTRVALUE.oneofs_by_name['value'].fields.append(
  _ATTRVALUE.fields_by_name['placeholder'])
_ATTRVALUE.fields_by_name['placeholder'].containing_oneof = _ATTRVALUE.oneofs_by_name['value']
_NAMEATTRLIST_ATTRENTRY.fields_by_name['value'].message_type = _ATTRVALUE
_NAMEATTRLIST_ATTRENTRY.containing_type = _NAMEATTRLIST
_NAMEATTRLIST.fields_by_name['attr'].message_type = _NAMEATTRLIST_ATTRENTRY

DESCRIPTOR.message_types_by_name['AttrValue'] = _ATTRVALUE
DESCRIPTOR.message_types_by_name['NameAttrList'] = _NAMEATTRLIST
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

AttrValue = _reflection.GeneratedProtocolMessageType('AttrValue', (_message.Message,), dict(
  ListValue = _reflection.GeneratedProtocolMessageType('ListValue', (_message.Message,), dict(
    DESCRIPTOR = _ATTRVALUE_LISTVALUE,
    __module__ = 'tensorboardX.proto.attr_value_pb2'
    ))
  ,
  DESCRIPTOR = _ATTRVALUE,
  __module__ = 'tensorboardX.proto.attr_value_pb2'
  ))
_sym_db.RegisterMessage(AttrValue)
_sym_db.RegisterMessage(AttrValue.ListValue)

NameAttrList = _reflection.GeneratedProtocolMessageType('NameAttrList', (_message.Message,), dict(
  AttrEntry = _reflection.GeneratedProtocolMessageType('AttrEntry', (_message.Message,), dict(
    DESCRIPTOR = _NAMEATTRLIST_ATTRENTRY,
    __module__ = 'tensorboardX.proto.attr_value_pb2'
    ))
  ,
  DESCRIPTOR = _NAMEATTRLIST,
  __module__ = 'tensorboardX.proto.attr_value_pb2'
  ))
_sym_db.RegisterMessage(NameAttrList)
_sym_db.RegisterMessage(NameAttrList.AttrEntry)

DESCRIPTOR._options = None
_ATTRVALUE_LISTVALUE.fields_by_name['i']._options = None
_ATTRVALUE_LISTVALUE.fields_by_name['f']._options = None
_ATTRVALUE_LISTVALUE.fields_by_name['b']._options = None
_ATTRVALUE_LISTVALUE.fields_by_name['type']._options = None
_NAMEATTRLIST_ATTRENTRY._options = None