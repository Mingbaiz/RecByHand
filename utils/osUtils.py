import json

def readTriple(path,sep=None):
    # with语句的作用：自动管理文件资源
    # path是路径 ‘r’是read表示只读模式  utf-8是文件的编码格式
    # 文件对象被赋值给f
    with open(path,'r',encoding='utf-8') as f:
        # readlines()方法会一次性读取文件中所有的行
        # 并将每行内容作为一个字符串元素储存在一个列表中
        for line in f.readlines():
            if sep:
                # .strip()去除字符串两端的空白字符
                # .split()按照分隔符将字符串拆分成列表
                lines = line.strip().split(sep)
            else:
                lines=line.strip().split()
            # 如果长度不是3那就跳过本次循环
            if len(lines)!=3:continue
            # 长度是3 利用yield将Lines列表作为一个生产的数据项返回
            yield lines

def readFile(path,sep=None):
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            if sep:
                lines = line.strip().split(sep)
            else:
                lines = line.strip().split()
            if len(lines)==0:continue
            yield lines

def getJson(path):
    with open(path,'r',encoding='utf-8') as f:
        d=json.load(f)
    return d

def dumpJson(obj,path):
    with open(path,'w+',encoding='utf-8') as f:
        json.dump(obj,f)