

def execute():
    string: str = "hello"
    print(list(string.encode("utf-8")))

    test_string = "hello! こんにちは!"
    utf8_encoded = test_string.encode("utf-8")
    print(utf8_encoded)

    print(chr(147))

    print(type(utf8_encoded))

    print(list(utf8_encoded))

    print(len(test_string))

    print(len(utf8_encoded))

    print(utf8_encoded.decode("utf-8"))



if __name__ == "__main__":
    execute()