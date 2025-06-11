class BaseSettings:
    def __init__(self, **data):
        for k, v in data.items():
            setattr(self, k, v)

    class Config:
        env_file = ""
        env_file_encoding = "utf-8"
        extra = "allow"


def Field(default=None, default_factory=None, **kwargs):
    if default is not None:
        return default
    if default_factory is not None:
        return default_factory()
    return None
