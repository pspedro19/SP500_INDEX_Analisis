class BaseSettings:
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)


def Field(default=None, *, env=None, default_factory=None):
    if default_factory is not None:
        return default_factory()
    return default
