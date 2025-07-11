import sys
b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorboardX/proto/plugin_mesh.proto',
  package='tensorboardX.mesh',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n$tensorboardX/proto/plugin_mesh.proto\x12\x11tensorboardX.mesh\"\xd7\x01\n\x0eMeshPluginData\x12\x0f\n\x07version\x18\x01 \x01(\x05\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x43\n\x0c\x63ontent_type\x18\x03 \x01(\x0e\x32-.tensorboardX.mesh.MeshPluginData.ContentType\x12\x13\n\x0bjson_config\x18\x05 \x01(\t\x12\r\n\x05shape\x18\x06 \x03(\x05\"=\n\x0b\x43ontentType\x12\r\n\tUNDEFINED\x10\x00\x12\n\n\x06VERTEX\x10\x01\x12\x08\n\x04\x46\x41\x43\x45\x10\x02\x12\t\n\x05\x43OLOR\x10\x03\x62\x06proto3'))
_MESHPLUGINDATA_CONTENTTYPE = _descriptor.EnumDescriptor(
  name='ContentType',
  full_name='tensorboardX.mesh.MeshPluginData.ContentType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNDEFINED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VERTEX', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FACE', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='COLOR', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=214,
  serialized_end=275,)
_sym_db.RegisterEnumDescriptor(_MESHPLUGINDATA_CONTENTTYPE)
_MESHPLUGINDATA = _descriptor.Descriptor(
  name='MeshPluginData',
  full_name='tensorboardX.mesh.MeshPluginData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='version', full_name='tensorboardX.mesh.MeshPluginData.version', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='tensorboardX.mesh.MeshPluginData.name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='content_type', full_name='tensorboardX.mesh.MeshPluginData.content_type', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='json_config', full_name='tensorboardX.mesh.MeshPluginData.json_config', index=3,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='shape', full_name='tensorboardX.mesh.MeshPluginData.shape', index=4,
      number=6, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _MESHPLUGINDATA_CONTENTTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=60,
  serialized_end=275,)
_MESHPLUGINDATA.fields_by_name['content_type'].enum_type = _MESHPLUGINDATA_CONTENTTYPE
_MESHPLUGINDATA_CONTENTTYPE.containing_type = _MESHPLUGINDATA
DESCRIPTOR.message_types_by_name['MeshPluginData'] = _MESHPLUGINDATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)
MeshPluginData = _reflection.GeneratedProtocolMessageType('MeshPluginData', (_message.Message,), dict(
  DESCRIPTOR = _MESHPLUGINDATA,
  __module__ = 'tensorboardX.proto.plugin_mesh_pb2'
))
_sym_db.RegisterMessage(MeshPluginData)