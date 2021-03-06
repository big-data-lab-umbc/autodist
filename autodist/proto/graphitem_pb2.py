# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: autodist/proto/graphitem.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='autodist/proto/graphitem.proto',
  package='autodist.proto',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x1e\x61utodist/proto/graphitem.proto\x12\x0e\x61utodist.proto\x1a\x19google/protobuf/any.proto\"\xd8\x02\n\tGraphItem\x12\'\n\tgraph_def\x18\x01 \x01(\x0b\x32\x14.google.protobuf.Any\x12I\n\x11grad_target_pairs\x18\x02 \x03(\x0b\x32..autodist.proto.GraphItem.GradTargetPairsEntry\x12,\n\x04info\x18\x03 \x01(\x0b\x32\x1e.autodist.proto.GraphItem.Info\x1a\x36\n\x14GradTargetPairsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x1aq\n\x04Info\x12\'\n\tvariables\x18\x01 \x03(\x0b\x32\x14.google.protobuf.Any\x12\x1a\n\x12table_initializers\x18\x02 \x03(\t\x12$\n\x06savers\x18\x03 \x03(\x0b\x32\x14.google.protobuf.Anyb\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_any__pb2.DESCRIPTOR,])




_GRAPHITEM_GRADTARGETPAIRSENTRY = _descriptor.Descriptor(
  name='GradTargetPairsEntry',
  full_name='autodist.proto.GraphItem.GradTargetPairsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='autodist.proto.GraphItem.GradTargetPairsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='autodist.proto.GraphItem.GradTargetPairsEntry.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=253,
  serialized_end=307,
)

_GRAPHITEM_INFO = _descriptor.Descriptor(
  name='Info',
  full_name='autodist.proto.GraphItem.Info',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='variables', full_name='autodist.proto.GraphItem.Info.variables', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='table_initializers', full_name='autodist.proto.GraphItem.Info.table_initializers', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='savers', full_name='autodist.proto.GraphItem.Info.savers', index=2,
      number=3, type=11, cpp_type=10, label=3,
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
  serialized_start=309,
  serialized_end=422,
)

_GRAPHITEM = _descriptor.Descriptor(
  name='GraphItem',
  full_name='autodist.proto.GraphItem',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='graph_def', full_name='autodist.proto.GraphItem.graph_def', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='grad_target_pairs', full_name='autodist.proto.GraphItem.grad_target_pairs', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='info', full_name='autodist.proto.GraphItem.info', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_GRAPHITEM_GRADTARGETPAIRSENTRY, _GRAPHITEM_INFO, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=78,
  serialized_end=422,
)

_GRAPHITEM_GRADTARGETPAIRSENTRY.containing_type = _GRAPHITEM
_GRAPHITEM_INFO.fields_by_name['variables'].message_type = google_dot_protobuf_dot_any__pb2._ANY
_GRAPHITEM_INFO.fields_by_name['savers'].message_type = google_dot_protobuf_dot_any__pb2._ANY
_GRAPHITEM_INFO.containing_type = _GRAPHITEM
_GRAPHITEM.fields_by_name['graph_def'].message_type = google_dot_protobuf_dot_any__pb2._ANY
_GRAPHITEM.fields_by_name['grad_target_pairs'].message_type = _GRAPHITEM_GRADTARGETPAIRSENTRY
_GRAPHITEM.fields_by_name['info'].message_type = _GRAPHITEM_INFO
DESCRIPTOR.message_types_by_name['GraphItem'] = _GRAPHITEM
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

GraphItem = _reflection.GeneratedProtocolMessageType('GraphItem', (_message.Message,), {

  'GradTargetPairsEntry' : _reflection.GeneratedProtocolMessageType('GradTargetPairsEntry', (_message.Message,), {
    'DESCRIPTOR' : _GRAPHITEM_GRADTARGETPAIRSENTRY,
    '__module__' : 'autodist.proto.graphitem_pb2'
    # @@protoc_insertion_point(class_scope:autodist.proto.GraphItem.GradTargetPairsEntry)
    })
  ,

  'Info' : _reflection.GeneratedProtocolMessageType('Info', (_message.Message,), {
    'DESCRIPTOR' : _GRAPHITEM_INFO,
    '__module__' : 'autodist.proto.graphitem_pb2'
    # @@protoc_insertion_point(class_scope:autodist.proto.GraphItem.Info)
    })
  ,
  'DESCRIPTOR' : _GRAPHITEM,
  '__module__' : 'autodist.proto.graphitem_pb2'
  # @@protoc_insertion_point(class_scope:autodist.proto.GraphItem)
  })
_sym_db.RegisterMessage(GraphItem)
_sym_db.RegisterMessage(GraphItem.GradTargetPairsEntry)
_sym_db.RegisterMessage(GraphItem.Info)


_GRAPHITEM_GRADTARGETPAIRSENTRY._options = None
# @@protoc_insertion_point(module_scope)
