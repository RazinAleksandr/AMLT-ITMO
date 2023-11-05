# Decorator for making the TS have only relevant features (that we used in anomalies research)
def preprocess_decorator(func):
    def wrapper(self, *args, **kwargs):
        df = func(self, *args, **kwargs)
        return self.preprocess_data(df)
    return wrapper