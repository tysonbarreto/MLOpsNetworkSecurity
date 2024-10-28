import logging
from logging import StreamHandler, FileHandler
import os
from datetime import datetime
import sys
from pathlib import Path

def logger():
    LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y')}.log"

    logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
    os.makedirs(logs_path,exist_ok=True)

    LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)


    stream_handler = StreamHandler(sys.stdout)
    file_handler = FileHandler(LOG_FILE_PATH)

    logging.basicConfig(
        #filename=LOG_FILE_PATH,
        format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            stream_handler,
            file_handler
        ]
    )

    return logging.getLogger('NSLogger')

if __name__=="__main__":
    __all__=["logger"]



