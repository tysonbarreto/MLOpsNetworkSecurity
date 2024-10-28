from functools import wraps
from loggings.NSlogger import logger
from exception.NSException import NSException
import sys

def TryExceptLogger(func):
    @wraps(func)
    def _wrapper(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except Exception as e:
            logger().info(NSException(e,sys))
    return _wrapper
if __name__=="__main__":
    __all__=["TryExceptLogger"]