DESC='''
Some tools for displays (for now).

By: AA
'''

def pf(func):
    def wrapper(*args, **kwargs):
        print('--------------------------------------------------')
        print(func.__name__)
        return func(*args, **kwargs)
    return wrapper
