# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: autodist/proto/strategy.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from autodist.proto import synchronizers_pb2 as autodist_dot_proto_dot_synchronizers__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='autodist/proto/strategy.proto',
  package='autodist.proto',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\x1d\x61utodist/proto/strategy.proto\x12\x0e\x61utodist.proto\x1a\"autodist/proto/synchronizers.proto\"\xab\x03\n\x08Strategy\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04path\x18\x02 \x01(\t\x12\x32\n\x0bnode_config\x18\x03 \x03(\x0b\x32\x1d.autodist.proto.Strategy.Node\x12:\n\x0cgraph_config\x18\x04 \x01(\x0b\x32$.autodist.proto.Strategy.GraphConfig\x1a\xf3\x01\n\x04Node\x12\x10\n\x08var_name\x18\x01 \x01(\t\x12\x38\n\x0ePSSynchronizer\x18\x02 \x01(\x0b\x32\x1e.autodist.proto.PSSynchronizerH\x00\x12\x46\n\x15\x41llReduceSynchronizer\x18\x03 \x01(\x0b\x32%.autodist.proto.AllReduceSynchronizerH\x00\x12\x13\n\x0bpartitioner\x18\x04 \x01(\t\x12\x32\n\x0bpart_config\x18\x05 \x03(\x0b\x32\x1d.autodist.proto.Strategy.NodeB\x0e\n\x0csynchronizer\x1a\x1f\n\x0bGraphConfig\x12\x10\n\x08replicas\x18\x01 \x03(\tb\x06proto3'
  ,
  dependencies=[autodist_dot_proto_dot_synchronizers__pb2.DESCRIPTOR,])




_STRATEGY_NODE = _descriptor.Descriptor(
  name='Node',
  full_name='autodist.proto.Strategy.Node',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='var_name', full_name='autodist.proto.Strategy.Node.var_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='PSSynchronizer', full_name='autodist.proto.Strategy.Node.PSSynchronizer', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='AllReduceSynchronizer', full_name='autodist.proto.Strategy.Node.AllReduceSynchronizer', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='partitioner', full_name='autodist.proto.Strategy.Node.partitioner', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='part_config', full_name='autodist.proto.Strategy.Node.part_config', index=4,
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
    _descriptor.OneofDescriptor(
      name='synchronizer', full_name='autodist.proto.Strategy.Node.synchronizer',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=237,
  serialized_end=480,
)

_STRATEGY_GRAPHCONFIG = _descriptor.Descriptor(
  name='GraphConfig',
  full_name='autodist.proto.Strategy.GraphConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='replicas', full_name='autodist.proto.Strategy.GraphConfig.replicas', index=0,
      number=1, type=9, cpp_type=9, label=3,
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
  serialized_start=482,
  serialized_end=513,
)

_STRATEGY = _descriptor.Descriptor(
  name='Strategy',
  full_name='autodist.proto.Strategy',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='autodist.proto.Strategy.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='path', full_name='autodist.proto.Strategy.path', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='node_config', full_name='autodist.proto.Strategy.node_config', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='graph_config', full_name='autodist.proto.Strategy.graph_config', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_STRATEGY_NODE, _STRATEGY_GRAPHCONFIG, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=86,
  serialized_end=513,
)

_STRATEGY_NODE.fields_by_name['PSSynchronizer'].message_type = autodist_dot_proto_dot_synchronizers__pb2._PSSYNCHRONIZER
_STRATEGY_NODE.fields_by_name['AllReduceSynchronizer'].message_type = autodist_dot_proto_dot_synchronizers__pb2._ALLREDUCESYNCHRONIZER
_STRATEGY_NODE.fields_by_name['part_config'].message_type = _STRATEGY_NODE
_STRATEGY_NODE.containing_type = _STRATEGY
_STRATEGY_NODE.oneofs_by_name['synchronizer'].fields.append(
  _STRATEGY_NODE.fields_by_name['PSSynchronizer'])
_STRATEGY_NODE.fields_by_name['PSSynchronizer'].containing_oneof = _STRATEGY_NODE.oneofs_by_name['synchronizer']
_STRATEGY_NODE.oneofs_by_name['synchronizer'].fields.append(
  _STRATEGY_NODE.fields_by_name['AllReduceSynchronizer'])
_STRATEGY_NODE.fields_by_name['AllReduceSynchronizer'].containing_oneof = _STRATEGY_NODE.oneofs_by_name['synchronizer']
_STRATEGY_GRAPHCONFIG.containing_type = _STRATEGY
_STRATEGY.fields_by_name['node_config'].message_type = _STRATEGY_NODE
_STRATEGY.fields_by_name['graph_config'].message_type = _STRATEGY_GRAPHCONFIG
DESCRIPTOR.message_types_by_name['Strategy'] = _STRATEGY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Strategy = _reflection.GeneratedProtocolMessageType('Strategy', (_message.Message,), {

  'Node' : _reflection.GeneratedProtocolMessageType('Node', (_message.Message,), {
    'DESCRIPTOR' : _STRATEGY_NODE,
    '__module__' : 'autodist.proto.strategy_pb2'
    # @@protoc_insertion_point(class_scope:autodist.proto.Strategy.Node)
    })
  ,

  'GraphConfig' : _reflection.GeneratedProtocolMessageType('GraphConfig', (_message.Message,), {
    'DESCRIPTOR' : _STRATEGY_GRAPHCONFIG,
    '__module__' : 'autodist.proto.strategy_pb2'
    # @@protoc_insertion_point(class_scope:autodist.proto.Strategy.GraphConfig)
    })
  ,
  'DESCRIPTOR' : _STRATEGY,
  '__module__' : 'autodist.proto.strategy_pb2'
  # @@protoc_insertion_point(class_scope:autodist.proto.Strategy)
  })
_sym_db.RegisterMessage(Strategy)
_sym_db.RegisterMessage(Strategy.Node)
_sym_db.RegisterMessage(Strategy.GraphConfig)


# @@protoc_insertion_point(module_scope)