import hashlib
import json
from dataclasses import dataclass
from qm import QuantumMachinesManager

__all__ = ["QMW"]

@dataclass
class QMW:
    qmm: QuantumMachinesManager
    qm_id: int
    close: bool = True

    def save_id_to_file(self, qm_id, filename):
        with open(filename, 'w') as file:
            file.write(str(qm_id))


    def load_id_from_file(self, filename):
        try:
            with open(filename, 'r') as file:
                return file.read().strip()  # Strip to remove extra whitespace/newlines
        except FileNotFoundError:
            return None  # Return None if the file doesn't exist

    def generate_config_hash(self, config):
        # Convert the dictionary to a JSON string with sorted keys
        config_str = json.dumps(config, sort_keys=True)
        # Generate a hash using SHA256
        config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()
        return config_hash


    def open_qm(self, config):
        close = self.close
        qm_id = self.qm_id
        if qm_id is None:
            qm_id = self.generate_config_hash(config)

        filename = f'qm_id-{qm_id}.txt'

        qms = self.qmm.list_open_qms()
        saved_id = self.load_id_from_file(filename)
        if saved_id in qms:
            qm = self.qmm.get_qm(saved_id)
            if close:
                qm.close()
            else:
                return qm

        qm = self.qmm.open_qm(config, close_other_machines=False)
        self._last_server_id = qm.id
        self.save_id_to_file(qm.id, filename)

        return qm
