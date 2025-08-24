# Importing logger and custom exception
from src.logger import get_logger
from src.exception import CustomException

import sys

# Initialize logger
logger = get_logger(__name__)

# Function to divide two numbers
def divide_number(a, b):
    try:
        result = a / b
        logger.info("Dividing 2 numbers")
        return result
    except Exception as e:
        logger.error("Error occurred")
        raise CustomException("Custom Error: zero", sys)

if __name__ == "__main__":
    try:
        logger.info("Starting main program")
        divide_number(10, 2)
    except CustomException as ce:
        logger.error(str(ce))