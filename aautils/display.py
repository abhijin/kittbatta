DESC='''
Some tools for displays (for now).

By: AA
'''

# print function name
def pf(func):
    def wrapper(*args, **kwargs):
        print('--------------------------------------------------')
        print(func.__name__)
        return func(*args, **kwargs)
    return wrapper
