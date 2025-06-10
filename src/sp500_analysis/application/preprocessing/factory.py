from sp500_analysis.application.preprocessing.processors.investing import InvestingProcessor
from sp500_analysis.application.preprocessing.processors.fred import FredProcessor
from sp500_analysis.application.preprocessing.processors.eoe import EOEProcessor


class ProcessorFactory:
    """Return processor instances based on a string identifier."""

    @staticmethod
    def get_processor(name: str, *args, **kwargs):
        name = name.lower()
        if name == 'investing':
            return InvestingProcessor(*args, **kwargs)
        if name == 'fred':
            return FredProcessor(*args, **kwargs)
        if name == 'eoe':
            return EOEProcessor(*args, **kwargs)
        raise ValueError(f"Unknown processor: {name}")
