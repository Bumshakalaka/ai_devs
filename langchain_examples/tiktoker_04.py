from typing import Dict

from tiktoken import get_encoding


# Ta funkcja ma za zadanie estymować liczbę tokenów w przekazanych wiadomościach z uwzględnieniem encodera konkretnego modelu
def count_tokens(messages: list[Dict], model="gpt-3.5-turbo-0613") -> int:
    encoding = get_encoding("cl100k_base")

    if model in [
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    ]:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4
        tokens_per_name = -1
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return count_tokens(messages, "gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return count_tokens(messages, "gpt-4-0613")
    else:
        raise NotImplementedError(
            f"num_tokens_from_messages() is not implemented for model {model}. S"
            f"ee https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."
        )

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name

    num_tokens += 3
    return num_tokens


if __name__ == "__main__":
    messages = [{"role": "system", "content": "Hey, you!", "ai": "What's up!"}]
    print(count_tokens(messages, "gpt-4"))
    print(get_encoding("cl100k_base").encode(messages[0]["content"]))
