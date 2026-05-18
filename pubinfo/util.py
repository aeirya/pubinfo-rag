
def stringify(dataset):
    data = dataset.apply(lambda x: str(x), axis=1).to_list()
    data = '\n\n'.join(data)
    return data