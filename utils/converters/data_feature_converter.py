def convert_array(data, features, convert_function):
    for feature in features:
        data[feature] = data[feature].apply(convert_function)
    return data


def convert_array_with_axis(data, features, convert_function, axis):
    for feature in features:
        data[feature] = data.apply(convert_function, axis)
    return data


def convert_part(data, feature, feature_part, convert_function):
    data[feature_part] = data[feature].apply(convert_function)
    return data
