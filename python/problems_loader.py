import os
import pickle
import mk_problem

class ProblemsLoader(object):
    def __init__(self, filenames):
        # We sort so that the smallest file (grp0) will be first
        self.filenames = sorted(filenames)
        print(self.filenames)

        self.next_file_num = 0
        assert(self.has_next())

    def has_next(self):
        return self.next_file_num < len(self.filenames)

    def get_next(self):
        if not self.has_next():
            self.reset()
        filename = self.filenames[self.next_file_num]
        print("Loading %s..." % filename)
        with open(filename, 'rb') as f:
            problems = pickle.load(f)
        self.next_file_num += 1
        assert(len(problems) > 0)
        return problems, filename

    def reset(self):
        # Note: skip the short first files on subsequent passes
        self.next_file_num = 0

def init_problems_loader(dirname):
    return ProblemsLoader([dirname + "/" + f for f in os.listdir(dirname)])
