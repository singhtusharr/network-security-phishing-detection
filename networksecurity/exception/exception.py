"""Custom exception that captures WHERE a failure happened, not just what.

Wrapping every component's risky call in `try/except Exception as e: raise
NetworkSecurityException(e, sys)` means every error in the logs/traceback
reports its own file name and line number explicitly, instead of relying on
Python's default traceback (which gets harder to read once the error has
bubbled up through several pipeline stages).
"""
import sys


class NetworkSecurityException(Exception):
    def __init__(self, error_message: Exception, error_details: sys):
        self.error_message = str(error_message)
        _, _, exc_tb = error_details.exc_info()
        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return (
            f"Error occurred in python script [{self.file_name}] "
            f"line number [{self.lineno}] error message [{self.error_message}]"
        )
