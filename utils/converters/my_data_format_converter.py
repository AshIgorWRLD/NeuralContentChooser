import numpy as np


def get_director(data):
    for i in data:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# Если элементы списка находятся в датасете, то возвращаем не больше 3 элементов списка
def get_list(data):
    if isinstance(data, list):
        names = [i['name'] for i in data]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


# Избавляемся от ненужных пробелов и переводим все слова в нижний регистр
def clean_data(data):
    if isinstance(data, list):
        return [str.lower(i.replace(" ", "")) for i in data]
    else:
        if isinstance(data, str):
            return str.lower(data.replace(" ", ""))
        else:
            return ''
