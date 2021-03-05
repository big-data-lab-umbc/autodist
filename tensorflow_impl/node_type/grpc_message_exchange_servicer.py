import time
import grpc
from . import garfield_pb2
from . import garfield_pb2_grpc
from . import tools

class MessageExchangeServicer(garfield_pb2_grpc.MessageExchangeServicer):

    def __init__(self, model_weights):
        """
            args: 
                - model_weights: 
        """

        self.model_wieghts_history = [model_weights]
        self.gradients_history = []


    def GetModel(self, request, context):
        """Get the model weights of a specific iteration stored on the server."""
        iter = request.iter
        job = request.job
        req_id = request.req_id

        while iter >= len(self.model_wieghts_history):
            time.sleep(0.001)
        
        serialized_model = self.model_wieghts_history[iter].tobytes()
        #serialized_model = tools.weights_to_bytes(self.model_wieghts_history[iter])
        return garfield_pb2.Model(model=serialized_model,
                                  init=True,
                                  iter=iter)
        

    def SendModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetGradient(self, request, context):
        """Get the graidents of a specific iteration stored on the server."""
        iter = request.iter
        job = request.job
        req_id = request.req_id

        while iter >= len(self.gradients_history):
            time.sleep(0.001)
        
        #serialized_gradients = [tools.tensor_to_bytes(grads) for grads in self.gradients_history[iter]]
        serialized_gradients = self.gradients_history[iter].tobytes()
        return garfield_pb2.Gradients(gradients=serialized_gradients,
                                      iter=iter)

    def SendGradient(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')