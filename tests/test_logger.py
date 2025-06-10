from sp500_analysis.shared.logging.logger import configurar_logging
import logging

def test_configurar_logging_writes_file(tmp_path):
    log_file = tmp_path / "test.log"
    logging.getLogger().handlers.clear()
    configurar_logging(str(log_file))
    logging.info("test message")
    assert "test message" in log_file.read_text()
