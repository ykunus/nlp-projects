from typing import Generator, Iterable
from collections import Counter


def sorted_chars(s: str) -> list[str]:
    unique_s = set(s)
    return sorted(unique_s)


def gen_sentences(path: str) -> Generator[list[str], None, None]:
    with open(path, encoding='utf-8') as file:
        for line in file:
            if line.strip() != "" :
                words = line.strip("\n").split(" ")
                yield words

def n_most_frequent_tokens(sentences: Iterable[list[str]], n: int) -> list[str]:
    most_frequent = []
    if n < 0:
        raise ValueError(f"Value {n} for n is negative")
    elif n == 0:
        return most_frequent
    else:
        c = Counter()
        for sentence in sentences:
            c.update(sentence)
        # if len(c) != 0:

        for element in c.most_common(n):
            most_frequent.append(element[0])
        return most_frequent

def case_sarcastically(text: str) -> str:
    result = ""
    upper = False
    for c in text:
        if c.upper() != c.lower():
            if not upper:
                result+=c.lower()
                upper = True
            else:
                result+= c.upper()
                upper = False
        else:
            result += c
    return result
