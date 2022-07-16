from torch import Tensor, nn

__all__ = ["model_dict", "register_model", "unpack_feature", "pack_feature", "record_f"]

model_dict = {}


def register_model(cls):
    key = cls.__name__.lower()
    if key not in model_dict:
        model_dict[key] = cls
    elif model_dict[key] != cls:
        raise KeyError(f"Duplicated key {cls.__name__} from {model_dict[cls.__name__]} and {cls}!!!!")


def unpack_feature(x):
    if isinstance(x, tuple):
        f_list, x = x
    else:
        f_list = []
    return f_list, x


def pack_feature(f_list: list, x: Tensor, with_feature: bool = True, add: bool = True):
    if with_feature:
        if add:
            return f_list + [x], x
        else:
            return f_list, x
    return x


def record_f(obj: nn.Module, default=True, record=True):
    def warpper(self, input, *inp, with_feature=default, **kwargs):
        f_list, x = unpack_feature(input)
        x = self.forward_(x, *inp, **kwargs)
        return pack_feature(f_list, x, with_feature=with_feature, add=record)

    from types import MethodType
    obj.forward_ = obj.forward
    obj.forward = MethodType(warpper, obj)
    return obj
