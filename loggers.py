import os
import sys

class LogStream:
    def __init__(self, path, stream):
        self.path = path
        self.stream = stream
        self.ready = False

    def write(self, x):
        if not self.ready:
            directory_name = os.path.dirname(self.path)
            os.makedirs(directory_name, exist_ok=True)
            self.ready = True
        with open(self.path, 'a+') as f_out:
            f_out.write(x)
        self.stream.write(x)

    def flush(self):
        self.stream.flush()


class LogAlreadyExistsError(Exception):
    def __init__(self, message):
        self.message = message


class Logger:
    def __init__(self, path):
        if os.path.exists(path):
            raise LogAlreadyExistsError('The log file {} already exists'.format(path))
        self.__stdout = LogStream(path, sys.stdout)
        self.__stderr = LogStream(path, sys.stderr)

    def connect(self):
        sys.stdout = self.__stdout
        sys.stderr = self.__stderr

    def disconnect(self):
        sys.stdout = self.__stdout.stream
        sys.stderr = self.__stderr.stream

