import numpy as np
import tensorflow as tf
import time
import sys
import grpc
from . import garfield_pb2_grpc

def bytes_to_chunks(s, size=4194290):
    """ Take bytes as argument and split them in diffenrent chunks
    
        args: 
            - s: bytes
        
        returns: list of bytes
    """
    return [s[i:i+size] for i in range(0, len(s), size)]



def model_to_bytes(model):
    """ Serialize the model parameters into a stream of bytes.
    
        args: 
            - model: keras model
        
        returns: list of bytes ready to be sent as a stream
    """
    parameters = model.get_weights()
    #chunks = [bytes_to_chunks(p.tobytes()) for p in parameters]
    chunks = [p.tobytes() for p in parameters]
    return chunks

def weights_to_bytes(weights):
    """ Serialize the model parameters into a bytes.
    
        args: 
            - weights: list of numpy array
        
        returns: 
            list of bytes representing layer's weights
    """
    #chunks = [bytes_to_chunks(p.tobytes()) for p in parameters]
    serialized_weights = [p.numpy().tobytes() for p in weights]
    return serialized_weights

def bytes_to_weights(bytes_list):
    """ Deserialize the model parameters from bytes.
    
        args: 
            - bytes: list of bytes
        
        returns: 
            list of numpy array containing the layer's flatten weights
    """
    #chunks = [bytes_to_chunks(p.tobytes()) for p in parameters]
    flatten_weights = [np.frombuffer(b, dtype=np.float32) for b in bytes_list]
    return flatten_weights

def tensor_to_bytes(tensor):
    """ Serialize a TensorFlow Tensor into bytes.

        args: 
            - Tensor
        returns:
            bytes
    """
    return tf.io.serialize_tensor(tensor).numpy()

def bytes_to_model(model, serialized_weights):
    """ Take the list of bytes of the parameters to recreate a Keras model.
    
        args: 
            - model: Keras model
            - serialized_parameters: list of bytes
              
        returns: Keras model
    """
    model_shape = [p.shape for p in model.get_weights()]

    weights = [np.frombuffer(w, dtype=np.float32).reshape(shape) for w, shape in zip(serialized_weights, model_shape)]
    model.set_weights(weights)
    return model

def flatten_weights(model):
    """ Flatten the model.

        args:
            - model: Keras model trainable variables
        
        returns:
            numpy array of all the weights.
    """
    return np.concatenate([l.numpy().reshape((-1,)) for l in model])

def reshape_weights(model, flatten_weights):
    """ Reshape the model from flatten numpy array.

        args:
            - model: Keras model
            - flatten_weights: numpy array
        
        returns:
            list of numpy array.
    """
    model_weights = []
    i=0
    for var in model.trainable_variables:
        shape = var.shape
        len_ = 1
        for j in shape:
            len_ *= j

        layer = np.array(flatten_weights[i:i+len_]).reshape(shape)
        model_weights.append(layer)
        i += len_
    return model_weights

def set_connection(host):
    """ Set the connection of a specific server.

        args:
            - hosts: string of the form "ip:port"

        returns:
            Stub
    """

    channel = grpc.insecure_channel(host, options=[('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)])
    stub = garfield_pb2_grpc.MessageExchangeStub(channel)
    
    return stub


def training_progression(MAX, iter, accuracy):
    char = chr(0x2588)
    i = int(iter/(MAX-1)*20)
    sys.stdout.write(f"\rTraining |{char * i}" + f"." * (20 - i) + f" | iter: {str(iter)}/{str(MAX-1)} Accuracy: {accuracy:.2f}%")
    sys.stdout.flush()