
def test_function(S):
    upper_list = []
    other_list = []
    for ch in S:
        if ch.isupper():
            upper_list.append(ch)
        else:
            other_list.append(ch)
    print(upper_list)
    print(other_list)
    return "".join(other_list+upper_list)



if __name__ == '__main__':
    test_string = "Alps-Electronic"
    result = test_function(test_string)
    print(result)

