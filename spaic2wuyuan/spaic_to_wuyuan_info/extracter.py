from typing import Callable


exts: dict[str, Callable] = {}
'''存储不同类的 get_info 函数，用来获取特定信息'''

vars: dict[str, dict[str, tuple[str, Callable]]] = {}
'''存储不同类的 var_dict 字典，存储 spaic 变量名的物源名称和形变函数'''


class Meta(type):
    def __new__(cls, name, *args):
        a = type.__new__(cls, name, *args)
        get_info = getattr(a, 'get_info', None)
        if get_info is not None:
            exts[name] = get_info
        var_dict = getattr(a, 'var_dict', None)
        if var_dict is not None:
            vars[name] = var_dict
        return a


class Extracter(metaclass=Meta):
    '''继承该类会自动注册 get_info 和 var_dict 两种属性'''
    pass


def update_info(info: dict, name: str, *args):
    '''根据类名解析并更新信息'''

    get_info = exts.get(name)
    if get_info is None:
        raise ValueError(f'没有 {name} 模型')

    info1 = get_info(*args)

    # dfs
    stack = [(info, info1)]
    while stack:
        d, d1 = stack.pop()
        for k, v1 in d1.items():
            v = d.get(k)
            if isinstance(v, dict) and isinstance(v1, dict):
                stack.append((v, v1))
            else:
                d[k] = v1
