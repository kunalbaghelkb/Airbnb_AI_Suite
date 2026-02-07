import sys
from src.logger import logging

def error_message_detail(error, error_detail:sys):
    '''
    This Function will give:
    1. In which file?
    2. On Which Line?
    3. What was the error message?
    '''
    # exc_info() it gives all details of execution
    _, _, exc_tb = error_detail.exc_info()
    
    # Get file name
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Format Error Message
    error_message = "Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        # Inherit Parent Exception
        super().__init__(error_message)
        # Generated Custom Message
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        
    def __str__(self):
        
        return self.error_message