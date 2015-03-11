class memoize:
    def __init__(self, function):
        self.function = function
        self.memoized = {}

    def __call__(self, *arglist):
        try:
            return self.memoized[arglist]
        except KeyError:
            self.memoized[arglist]= self.function(*arglist)
            return self.memoized[arglist]

