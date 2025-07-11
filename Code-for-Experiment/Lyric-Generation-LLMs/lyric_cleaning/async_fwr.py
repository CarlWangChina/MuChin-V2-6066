import threading
from os.path import basename

class FwThread(threading.Thread):
    def __init__(self, fname, fcont, wmode='w'):
        super().__init__(name=f'fwt-4-{basename(fname)}')
        self.fname = fname
        self.fcont = fcont
        self.wmode = wmode

    def run(self):
        with open(self.fname, self.wmode, encoding="utf-8") as f:
            f.write(self.fcont)