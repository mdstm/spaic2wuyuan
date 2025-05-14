from time import time


def test(text: str):
    def f(func):
        print(f'开始 {text}')
        t = time()
        ret = func()
        t = time() - t
        print(f'{text} 完成，用时 {t} 秒')
        return ret
    return f
