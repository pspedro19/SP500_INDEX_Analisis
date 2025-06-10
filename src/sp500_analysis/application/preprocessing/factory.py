from sp500_analysis.application.preprocessing.processors.investing import InvestingProcessor
from sp500_analysis.application.preprocessing.processors.fred import FredProcessor
from sp500_analysis.application.preprocessing.processors.eoe import EOEProcessor
from sp500_analysis.application.preprocessing.processors.economic import EconomicDataProcessor
from sp500_analysis.application.preprocessing.processors.fred_data import FredDataProcessor
from sp500_analysis.application.preprocessing.processors.banco_republica import BancoRepublicaProcessor


class ProcessorFactory:
    """Return processor instances based on a string identifier."""

    @staticmethod
    def get_processor(name: str, *args, **kwargs):
        name = name.lower()
        if name == 'investing':
            return InvestingProcessor(*args, **kwargs)
        if name in {'economic', 'economicdata'}:
            return EconomicDataProcessor(*args, **kwargs)
        if name in {'fred', 'freddata'}:
            return FredProcessor(*args, **kwargs)
        if name == 'fred_data_processor':
            return FredDataProcessor(*args, **kwargs)
        if name == 'banco_republica':
            return BancoRepublicaProcessor(*args, **kwargs)
        if name == 'eoe':
            return EOEProcessor(*args, **kwargs)
        raise ValueError(f"Unknown processor: {name}")
