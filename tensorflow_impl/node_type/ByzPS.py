from .PS import PS
from attacks.attacker import Attacker


class ByzWorker(PS):
    """ Class defining a byzantine parameter server. """

    def __init__(self, network=None, log=False, asyncr=False, dataset="mnist", max_episods=20, batch_size=128, nb_byz_worker=0, native=False):
        """ Create a Byzantine parameter server node.

            args:
                - network:  State of the cluster
                - log:      Boolean indicating whether to log or not
                - asyncr:   Boolean

        """
        super().__init__(network, log, asyncr, dataset, max_episods, batch_size, nb_byz_worker, native)
        self.attacker = Attacker(network.get_my_attack())

    def commit_model(self, **kwargs):
        """ Make the modification of the model available on the network. 

            Args:
                - kwargs:   Arguments needed for the attack
        """
        attacked_parameters = self.attacker.attack(tools.flatten_weights(self.model.trainable_variables), kwargs)
        self.service.model_wieghts_history.append(attacked_parameters)
