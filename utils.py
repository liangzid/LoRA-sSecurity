"""
======================================================================
UTILS --- 

    Author: Zi Liang <zi1415926.liang@connect.polyu.hk>
    Copyright © 2025, ZiLiang, all rights reserved.
    Created: 25 March 2025
======================================================================
"""


# ------------------------ Code --------------------------------------
import random
from openai import OpenAI
import json
import os


def random_insert(target, to_insert):
    result = target.copy()

    insert_positions = sorted(
        [random.randint(0, len(result)) for _ in to_insert])

    for pos, item in zip(insert_positions, to_insert):
        result.insert(pos, item)

    return result


apikey = os.environ["DEEPSEEK_API_KEY"]

client = OpenAI(
    api_key=apikey,
    base_url="https://api.deepseek.com",
)


def onetimequery(
    system_prompt,
    user_prompt,
):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )
    text = response.choices[0].message.content
    return text


def randomtake(als, num=1):
    if len(als) < num:
        return als.copy()
    return random.sample(als, num)


def LLM_Rephrase(inp, style="poetry"):

    PROMPT = "Please rephrase User's sentence into a poetry style. Do not change the original meaning of User's sentences. Your response should only consist of the sentence after rephrase"
    resp = onetimequery(PROMPT, inp)

    return resp


def test_func():
    list1 = [1, 2, 3, 4, 5, 6]
    list2 = [-1, -2, -3]

    print(randomtake(list1, 2))
    print(randomtake(list1, 2))
    print(randomtake(list1, 2))
    print("---------------------")
    print(random_insert(list1, list2))
    print(random_insert(list1, list2))
    print(random_insert(list1, list2))

    print(LLM_Rephrase("What is the meaning of life?"))
    print("---------------------")
    print(LLM_Rephrase("I like summer. Do you like summer?"))
    pass


if __name__ == "__main__":
    test_func()
