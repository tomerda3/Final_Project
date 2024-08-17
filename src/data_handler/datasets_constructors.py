from src.engine import construct_HHD_engine, construct_KHATT_engine
from src.data.path_variables import *

constructors = {
    HHD: construct_HHD_engine,
    KHATT: construct_KHATT_engine
}