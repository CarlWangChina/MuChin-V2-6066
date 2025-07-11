import sys
_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
from tensorboardX.proto import summary_pb2 as tensorboardX_dot_proto_dot_summary__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorboardX/proto/event.proto',
  package='tensorboardX',
  syntax='proto3',
  serialized_options=_b('\n\023org.tensorflow.utilB\013EventProtosP\001\370\001\001'),
  serialized_pb=_b('\n\x1etensorboardX/proto/event.proto\x12\x0ctensorboardX\x1a tensorboardX/proto/summary.proto\"\xc3\x02\n\x05\x45vent\x12\x11\n\twall_time\x18\x01 \x01(\x01\x12\x0c\n\x04step\x18\x02 \x01(\x03\x12\x16\n\x0c\x66ile_version\x18\x03 \x01(\tH\x00\x12\x13\n\tgraph_def\x18\x04 \x01(\x0cH\x00\x12(\n\x07summary\x18\x05 \x01(\x0b\x32\x15.tensorboardX.SummaryH\x00\x12/\n\x0blog_message\x18\x06 \x01(\x0b\x32\x18.tensorboardX.LogMessageH\x00\x12/\n\x0bsession_log\x18\x07 \x01(\x0b\x32\x18.tensorboardX.SessionLogH\x00\x12>\n\x13tagged_run_metadata\x18\x08 \x01(\x0b\x32\x1f.tensorboardX.TaggedRunMetadataH\x00\x12\x18\n\x0emeta_graph_def\x18\t \x01(\x0cH\x00\x42\x06\n\x04what\"\x97\x01\n\nLogMessage\x12-\n\x05level\x18\x01 \x01(\x0e\x32\x1e.tensorboardX.LogMessage.Level\x12\x0f\n\x07message\x18\x02 \x01(\t\"I\n\x05Level\x12\x0b\n\x07UNKNOWN\x10\x00\x12\t\n\x05\x44\x45\x42UG\x10\n\x12\x08\n\x04INFO\x10\x14\x12\x08\n\x04WARN\x10\x1e\x12\t\n\x05\x45RROR\x10(\x12\t\n\x05\x46\x41TAL\x10\x32\"\xb8\x01\n\nSessionLog\x12\x36\n\x06status\x18\x01 \x01(\x0e\x32&.tensorboardX.SessionLog.SessionStatus\x12\x17\n\x0f\x63heckpoint_path\x18\x02 \x01(\t\x12\x0b\n\x03msg\x18\x03 \x01(\t\"L\n\rSessionStatus\x12\x16\n\x12STATUS_UNSPECIFIED\x10\x00\x12\t\n\x05START\x10\x01\x12\x08\n\x04STOP\x10\x02\x12\x0e\n\nCHECKPOINT\x10\x03\"6\n\x11TaggedRunMetadata\x12\x0b\n\x03tag\x18\x01 \x01(\t\x12\x14\n\x0crun_metadata\x18\x02 \x01(\x0c\x42\'\n\x13org.tensorflow.utilB\x0b\x45ventProtosP\x01\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[tensorboardX_dot_proto_dot_summary__pb2.DESCRIPTOR,])

_LOGMESSAGE_LEVEL = _descriptor.EnumDescriptor(
  name='Level',
  full_name='tensorboardX.LogMessage.Level',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DEBUG', index=1, number=10,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INFO', index=2, number=20,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='WARN', index=3, number=30,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ERROR', index=4, number=40,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FATAL', index=5, number=50,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=487,
  serialized_end=560,)
_sym_db.RegisterEnumDescriptor(_LOGMESSAGE_LEVEL)

_SESSIONLOG_SESSIONSTATUS = _descriptor.EnumDescriptor(
  name='SessionStatus',
  full_name='tensorboardX.SessionLog.SessionStatus',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='STATUS_UNSPECIFIED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='START', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STOP', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CHECKPOINT', index=3, number=3,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=671,
  serialized_end=747,)
_sym_db.RegisterEnumDescriptor(_SESSIONLOG_SESSIONSTATUS)

_EVENT = _descriptor.Descriptor(
  name='Event',
  full_name='tensorboardX.Event',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='wall_time', full_name='tensorboardX.Event.wall_time', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='step', full_name='tensorboardX.Event.step', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='file_version', full_name='tensorboardX.Event.file_version', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='graph_def', full_name='tensorboardX.Event.graph_def', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='summary', full_name='tensorboardX.Event.summary', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='log_message', full_name='tensorboardX.Event.log_message', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='session_log', full_name='tensorboardX.Event.session_log', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tagged_run_metadata', full_name='tensorboardX.Event.tagged_run_metadata', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='meta_graph_def', full_name='tensorboardX.Event.meta_graph_def', index=8,
      number=9, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
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
      name='what', full_name='tensorboardX.Event.what',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=83,
  serialized_end=406,)

_LOGMESSAGE = _descriptor.Descriptor(
  name='LogMessage',
  full_name='tensorboardX.LogMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='level', full_name='tensorboardX.LogMessage.level', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='message', full_name='tensorboardX.LogMessage.message', index=1,
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
    _LOGMESSAGE_LEVEL,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=409,
  serialized_end=560,)

_SESSIONLOG = _descriptor.Descriptor(
  name='SessionLog',
  full_name='tensorboardX.SessionLog',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='tensorboardX.SessionLog.status', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='checkpoint_path', full_name='tensorboardX.SessionLog.checkpoint_path', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='msg', full_name='tensorboardX.SessionLog.msg', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _SESSIONLOG_SESSIONSTATUS,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=563,
  serialized_end=747,)

_TAGGEDRUNMETADATA = _descriptor.Descriptor(
  name='TaggedRunMetadata',
  full_name='tensorboardX.TaggedRunMetadata',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='tag', full_name='tensorboardX.TaggedRunMetadata.tag', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='run_metadata', full_name='tensorboardX.TaggedRunMetadata.run_metadata', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
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
  serialized_start=749,
  serialized_end=803,)

_EVENT.fields_by_name['summary'].message_type = tensorboardX_dot_proto_dot_summary__pb2._SUMMARY
_EVENT.fields_by_name['log_message'].message_type = _LOGMESSAGE
_EVENT.fields_by_name['session_log'].message_type = _SESSIONLOG
_EVENT.fields_by_name['tagged_run_metadata'].message_type = _TAGGEDRUNMETADATA
_EVENT.oneofs_by_name['what'].fields.append(
  _EVENT.fields_by_name['file_version'])
_EVENT.fields_by_name['file_version'].containing_oneof = _EVENT.oneofs_by_name['what']
_EVENT.oneofs_by_name['what'].fields.append(
  _EVENT.fields_by_name['graph_def'])
_EVENT.fields_by_name['graph_def'].containing_oneof = _EVENT.oneofs_by_name['what']
_EVENT.oneofs_by_name['what'].fields.append(
  _EVENT.fields_by_name['summary'])
_EVENT.fields_by_name['summary'].containing_oneof = _EVENT.oneofs_by_name['what']
_EVENT.oneofs_by_name['what'].fields.append(
  _EVENT.fields_by_name['log_message'])
_EVENT.fields_by_name['log_message'].containing_oneof = _EVENT.oneofs_by_name['what']
_EVENT.oneofs_by_name['what'].fields.append(
  _EVENT.fields_by_name['session_log'])
_EVENT.fields_by_name['session_log'].containing_oneof = _EVENT.oneofs_by_name['what']
_EVENT.oneofs_by_name['what'].fields.append(
  _EVENT.fields_by_name['tagged_run_metadata'])
_EVENT.fields_by_name['tagged_run_metadata'].containing_oneof = _EVENT.oneofs_by_name['what']
_EVENT.oneofs_by_name['what'].fields.append(
  _EVENT.fields_by_name['meta_graph_def'])
_EVENT.fields_by_name['meta_graph_def'].containing_oneof = _EVENT.oneofs_by_name['what']
_LOGMESSAGE.fields_by_name['level'].enum_type = _LOGMESSAGE_LEVEL
_LOGMESSAGE_LEVEL.containing_type = _LOGMESSAGE
_SESSIONLOG.fields_by_name['status'].enum_type = _SESSIONLOG_SESSIONSTATUS
_SESSIONLOG_SESSIONSTATUS.containing_type = _SESSIONLOG

DESCRIPTOR.message_types_by_name['Event'] = _EVENT
DESCRIPTOR.message_types_by_name['LogMessage'] = _LOGMESSAGE
DESCRIPTOR.message_types_by_name['SessionLog'] = _SESSIONLOG
DESCRIPTOR.message_types_by_name['TaggedRunMetadata'] = _TAGGEDRUNMETADATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Event = _reflection.GeneratedProtocolMessageType('Event', (_message.Message,), dict(
  DESCRIPTOR = _EVENT,
  __module__ = 'tensorboardX.proto.event_pb2'
  ))
_sym_db.RegisterMessage(Event)

LogMessage = _reflection.GeneratedProtocolMessageType('LogMessage', (_message.Message,), dict(
  DESCRIPTOR = _LOGMESSAGE,
  __module__ = 'tensorboardX.proto.event_pb2'
  ))
_sym_db.RegisterMessage(LogMessage)

SessionLog = _reflection.GeneratedProtocolMessageType('SessionLog', (_message.Message,), dict(
  DESCRIPTOR = _SESSIONLOG,
  __module__ = 'tensorboardX.proto.event_pb2'
  ))
_sym_db.RegisterMessage(SessionLog)

TaggedRunMetadata = _reflection.GeneratedProtocolMessageType('TaggedRunMetadata', (_message.Message,), dict(
  DESCRIPTOR = _TAGGEDRUNMETADATA,
  __module__ = 'tensorboardX.proto.event_pb2'
  ))
_sym_db.RegisterMessage(TaggedRunMetadata)

DESCRIPTOR._options = None