import re

WORD_TO_NUM = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9"
}

def extract_coordinates(text: str):
    """
    Returns first two numbers found in text as (x,y)
    If <2 numbers found -> (None,None)
    """

    text = text.lower()

    # Replace number words with digits
    for word, digit in WORD_TO_NUM.items():
        text = re.sub(rf"\b{word}\b", digit, text)

    # Find all numbers
    numbers = re.findall(r"\b\d+\b", text)

    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])

    return None, None
