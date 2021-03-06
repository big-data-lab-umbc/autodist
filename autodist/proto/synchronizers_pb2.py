# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: autodist/proto/synchronizers.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='autodist/proto/synchronizers.proto',
  package='autodist.proto',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=b'\n\"autodist/proto/synchronizers.proto\x12\x0e\x61utodist.proto\"k\n\x0ePSSynchronizer\x12\x1d\n\x15reduction_destination\x18\x01 \x01(\t\x12\x19\n\x11local_replication\x18\x02 \x01(\x08\x12\x0c\n\x04sync\x18\x03 \x01(\x08\x12\x11\n\tstaleness\x18\x04 \x01(\x05\"\x9e\x02\n\x15\x41llReduceSynchronizer\x12\x38\n\x04spec\x18\x01 \x01(\x0e\x32*.autodist.proto.AllReduceSynchronizer.Spec\x12\x44\n\ncompressor\x18\x02 \x01(\x0e\x32\x30.autodist.proto.AllReduceSynchronizer.Compressor\x12\r\n\x05group\x18\x03 \x01(\x05\"$\n\x04Spec\x12\x08\n\x04\x41UTO\x10\x00\x12\x08\n\x04NCCL\x10\x01\x12\x08\n\x04RING\x10\x02\"P\n\nCompressor\x12\x12\n\x0eNoneCompressor\x10\x00\x12\x15\n\x11HorovodCompressor\x10\x01\x12\x17\n\x13HorovodCompressorEF\x10\x02\x62\x06proto3'
)



_ALLREDUCESYNCHRONIZER_SPEC = _descriptor.EnumDescriptor(
  name='Spec',
  full_name='autodist.proto.AllReduceSynchronizer.Spec',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='AUTO', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NCCL', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RING', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=332,
  serialized_end=368,
)
_sym_db.RegisterEnumDescriptor(_ALLREDUCESYNCHRONIZER_SPEC)

_ALLREDUCESYNCHRONIZER_COMPRESSOR = _descriptor.EnumDescriptor(
  name='Compressor',
  full_name='autodist.proto.AllReduceSynchronizer.Compressor',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NoneCompressor', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HorovodCompressor', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HorovodCompressorEF', index=2, number=2,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=370,
  serialized_end=450,
)
_sym_db.RegisterEnumDescriptor(_ALLREDUCESYNCHRONIZER_COMPRESSOR)


_PSSYNCHRONIZER = _descriptor.Descriptor(
  name='PSSynchronizer',
  full_name='autodist.proto.PSSynchronizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='reduction_destination', full_name='autodist.proto.PSSynchronizer.reduction_destination', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='local_replication', full_name='autodist.proto.PSSynchronizer.local_replication', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sync', full_name='autodist.proto.PSSynchronizer.sync', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='staleness', full_name='autodist.proto.PSSynchronizer.staleness', index=3,
      number=4, type=5, cpp_type=1, label=1,
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
  serialized_start=54,
  serialized_end=161,
)


_ALLREDUCESYNCHRONIZER = _descriptor.Descriptor(
  name='AllReduceSynchronizer',
  full_name='autodist.proto.AllReduceSynchronizer',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='spec', full_name='autodist.proto.AllReduceSynchronizer.spec', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='compressor', full_name='autodist.proto.AllReduceSynchronizer.compressor', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='group', full_name='autodist.proto.AllReduceSynchronizer.group', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _ALLREDUCESYNCHRONIZER_SPEC,
    _ALLREDUCESYNCHRONIZER_COMPRESSOR,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=164,
  serialized_end=450,
)

_ALLREDUCESYNCHRONIZER.fields_by_name['spec'].enum_type = _ALLREDUCESYNCHRONIZER_SPEC
_ALLREDUCESYNCHRONIZER.fields_by_name['compressor'].enum_type = _ALLREDUCESYNCHRONIZER_COMPRESSOR
_ALLREDUCESYNCHRONIZER_SPEC.containing_type = _ALLREDUCESYNCHRONIZER
_ALLREDUCESYNCHRONIZER_COMPRESSOR.containing_type = _ALLREDUCESYNCHRONIZER
DESCRIPTOR.message_types_by_name['PSSynchronizer'] = _PSSYNCHRONIZER
DESCRIPTOR.message_types_by_name['AllReduceSynchronizer'] = _ALLREDUCESYNCHRONIZER
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PSSynchronizer = _reflection.GeneratedProtocolMessageType('PSSynchronizer', (_message.Message,), {
  'DESCRIPTOR' : _PSSYNCHRONIZER,
  '__module__' : 'autodist.proto.synchronizers_pb2'
  # @@protoc_insertion_point(class_scope:autodist.proto.PSSynchronizer)
  })
_sym_db.RegisterMessage(PSSynchronizer)

AllReduceSynchronizer = _reflection.GeneratedProtocolMessageType('AllReduceSynchronizer', (_message.Message,), {
  'DESCRIPTOR' : _ALLREDUCESYNCHRONIZER,
  '__module__' : 'autodist.proto.synchronizers_pb2'
  # @@protoc_insertion_point(class_scope:autodist.proto.AllReduceSynchronizer)
  })
_sym_db.RegisterMessage(AllReduceSynchronizer)


# @@protoc_insertion_point(module_scope)
