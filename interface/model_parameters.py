class ModelParams:
    def __init__(self):
        pass

    def override(self, override_params):
        if not isinstance(override_params, dict):
            raise Exception(f"override_params must be dictionary but found {type(override_params)}")
        for k, v in override_params.items():
            if not hasattr(self, k):
                raise Exception(f"Found invalid parameter: {k}. Only support: {', '.join(self.__dict__.keys())}")
            setattr(self, k, v)

