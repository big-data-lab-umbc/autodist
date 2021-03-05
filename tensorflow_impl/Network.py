import json


class Network:

    def __init__(self, tf_location):
        with open(tf_location) as json_file:
            self._data = json.load(json_file)
            self._ps = []
            self._worker = []

            self._ps = self._data['cluster']['ps']
            print(self._ps)

            self._worker = self._data['cluster']['worker']
            print(self._worker)

    def get_task_type(self):
        return self._data['task']['type']

    def get_task_index(self):
        return self._data['task']['index']

    def get_my_strategy(self):
        return self._data['task']['strategy']

    def get_my_attack(self):
        return self._data['task']['attack']

    def get_all_ps(self):
        """
            Return all PSs.
            If my server is a PS as well, it is excluded from the list
        """
        all_ps = self._ps.copy()
        #if self.get_task_type() == 'ps':
        #    all_ps.remove(self.get_my_node())

        return all_ps

    def get_all_other_worker(self):
        """
            Return all workers.
            If my server is a worker as well, it is excluded from the list
        """
        all_worker = self._worker.copy()
        #if self.get_task_type() == 'worker':
        #    all_worker.remove(self.get_my_node())

        return all_worker

    def get_all_workers(self):
        return self._worker

    def get_my_node(self):
        index = self._data['task']['index']
        if self.get_task_type() == "ps":
            return self._ps[index]
        elif self.get_task_type() == "worker":
            return self._worker[index]

    def get_my_port(self):
        return self.get_my_node().split(':')[1]
