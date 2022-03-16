import time
from typing import TypeVar, Callable, cast
import functools


_F = TypeVar('_F', bound=Callable)


def timeit(func: _F) -> _F:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            end = time.perf_counter()
            span = end - start
            print(f'{func.__name__}: {span:.2f} s')
    return cast(_F, wrapper)
