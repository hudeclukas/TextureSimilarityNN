import json

class singleton(type):
    _instances = {}
    def __call__(self, *args, **kwargs):
        if self not in self._instances:
            self._instances[self] = super(singleton, self).__call__(*args, **kwargs)
        return self._instances[self]

class configuration(metaclass=singleton):
    __read = False

    def __init__(self, file=None):
        if file == None:
            return
        if not self.__read:
            self.__read = True
            f = open(file)
            self.__config = json.load(f)
            for key in self.__config.keys():
                self.__dict__[key] = self.__config[key]
                print(key,':',self.__config[key])

    def add(self, key, value):
        self.__dict__[key]=value

if __name__ == '__main__':
    cfig = configuration('config.json')
    print(cfig.name)