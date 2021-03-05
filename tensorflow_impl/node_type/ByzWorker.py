from .Worker import Worker
from attacks.attacker import Attacker
from . import tools

class ByzWorker(Worker):
    """ Class defining a byzantine worker. """

    def __init__(self, network=None, log=False, asyncr=False, dataset="mnist", model="Simple",  batch_size=128, nb_byz_worker=0, native=False):
        """ Create a Byzantine Worker node.

            args:
                - network:  State of the cluster
                - log:      Boolean indicating whether to log or not
                - asyncr:   Boolean

        """
        super().__init__(network, log, asyncr, dataset, model, batch_size, nb_byz_worker, native)
        self.attacker = Attacker(network.get_my_attack())

    def compute_gradients(self, iter, **kwargs):
        """ Compute a byzantine gradient. """

        loss, gradient = super().compute_gradients(iter)
        
        if self.network.get_my_attack() in ['LittleIsNotEnough', 'FallEmpires']:
            byz_gradients = [super().compute_gradients(iter+1+i)[1] for i in range(self.nb_byz_worker-1)]
            gradient = self.attacker.attack(gradient=gradient, byz_gradients=byz_gradients)
        else:
            gradient = self.attacker.attack(gradient=gradient)

        return loss, gradient