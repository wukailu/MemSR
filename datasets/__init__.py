from .dataProvider import DataProvider


def query_dataset(name: str):
    return dataProvider.DatasetFactory.analyze_name(name, {}, type_only=True)
