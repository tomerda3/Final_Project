from src.engine import construct_hhd_engine, construct_khatt_engine
from src.data.path_variables import *

constructors = {
    HHD: construct_hhd_engine,
    KHATT: construct_khatt_engine
}