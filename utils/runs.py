import pickle
import os

class Runs(object):
    # Pseudocode
        # If not initialised before, initialise it to 0
        # Else, increase run count by 1 whenever the script is run
    # Add functionality to alter run count
    # Add functionality with pickle
    def __init__(self):
        self.file_name = 'run_counts.pickle'
        if self._check_pickle(self.file_name):
            # Load pickle
            print('...run pickle loaded...')
            self.count = self._load(self.file_name)
        else:
            self.count = 0
            # Pickle count
            print('...run pickle saved')
            self._save(self.file_name, self.count)

    def _check_pickle(self, file_name):
        return os.path.exists(file_name)

    def _load(self, file_name):
        with open(file_name, 'rb') as pickle_in:
            count = pickle.load(pickle_in)
            return count

    def _save(self, file_name, count):
        with open(file_name, 'wb') as pickle_out:
            pickle.dump(count, pickle_out)

    def update_runs(self, count=None):
        if count:
            self.count = count
        else:
            self.count += 1
        self._save(self.file_name, self.count)