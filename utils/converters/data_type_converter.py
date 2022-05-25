def convert(data, column_type, converting_type):
    data[column_type] = data[column_type].astype(converting_type)
    return data
