from ..interface.model_parameters import ModelParams


class FXNxModelParams(ModelParams):
    def __init__(self, override_params=None):
        super(FXNxModelParams, self).__init__()
        self.sim_path = 3000
        self.calib_time_steps = 300
        self.pricing_time_steps = 300
        self.model_name = "DUPIRE"

        if override_params is not None:
            self.override(override_params)

    def __eq__(self, other):
        if not isinstance(other, FXNxModelParams):
            return False
        if self.sim_path != other.sim_path:
            return False
        if self.calib_time_steps != other.calib_time_steps:
            return False
        if self.pricing_time_steps != other.pricing_time_steps:
            return False
        return self.model_name == other.model_name

    def __hash__(self):
        return hash((self.sim_path, self.calib_time_steps, self.pricing_time_steps, self.model_name))


if __name__ == '__main__':
    param = FXNxModelParams({"sim_path": 200})
    print(param.sim_path, param.calib_time_steps, param.pricing_time_steps)
