class DummyWriter:
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def log_image(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def flush(self):
        pass

    def __getattr__(self, name):
        def dummy(*args, **kwargs):
            pass

        return dummy
