import os
def read_input():

    # 获取当前文件的路径
    current_path = os.path.abspath(__file__)
    # 获取当前文件的上一级路径
    parent_path = os.path.dirname(current_path)
    # 要获取的文件名
    file_name = "input.txt"

    # 构建要获取的文件的路径
    file_path = os.path.join(parent_path, file_name)

    input_data_list =[]
    with open(file_path, "r", encoding='utf-8') as f:
        # readlines()：读取文件全部内容，以列表形式返回结果
        data = f.readlines()
        for item in data:
            if item[-1] =="\n":
                item = item[:-1]

            index = item.index(":")
            input_data = item[index+1:]
            input_data_list.append(input_data)

    a = int(input_data_list[0])
    b = int(input_data_list[1])
    c = int(input_data_list[2])
    d = int(input_data_list[3])
    e = input_data_list[4]


    return a,b,c,d,e


if __name__ == '__main__':
    read_input()