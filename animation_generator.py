import os
from enum import Enum


class Resolution(str, Enum):
    LOW = "l"
    MEDIUM = "m"
    HIGH = "h"


if __name__ == "__main__":
    class_name = os.getenv("CLASS")
    file = os.getenv("FILE_TO_RUN")
    res = Resolution(os.getenv("RESOLUTION"))

    # os.system(f"manim -pq{res.value} {file} {class_name}")
    os.system(f"manim --disable_caching -pq{res.value} {file} {class_name}")
