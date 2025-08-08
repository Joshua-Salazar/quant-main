from ..interface.model_parameters import ModelParams


class EQNxModelParams(ModelParams):
    def __init__(self, override_params=None):
        super(EQNxModelParams, self).__init__()
        self.sim_path = 3000
        self.calib_time_steps = 300
        self.pricing_time_steps = 300
        self.x_steps = 200
        self.model_name = "DUPIRE"
        self.exercise_probability_threshold = 0.05
        self.random_number = "PSEUDORANDOM"
        self.mc_type = ""
        self.mc_discretization_scheme = "1ST ORDER"
        self.antithetic = False
        self.use_model_date = False
        self.processing_unit = ""
        self.seed = 0
        self.use_abs_strike = False
        self.use_ctp_config = True

        if override_params is not None:
            self.override(override_params)

    def __eq__(self, other):
        if not isinstance(other, EQNxModelParams):
            return False
        if self.sim_path != other.sim_path:
            return False
        if self.calib_time_steps != other.calib_time_steps:
            return False
        if self.pricing_time_steps != other.pricing_time_steps:
            return False
        if self.x_steps != other.x_steps:
            return False
        if self.model_name != other.model_name:
            return False
        if self.exercise_probability_threshold != other.exercise_probability_threshold:
            return False
        if self.random_number != other.random_number:
            return False
        if self.mc_type != other.mc_type:
            return False
        if self.mc_discretization_scheme != other.mc_discretization_scheme:
            return False
        if self.antithetic != other.antithetic:
            return False
        if self.use_model_date != other.use_model_date:
            return False
        if self.processing_unit != other.processing_unit:
            return False
        if self.seed != other.seed:
            return False
        if self.use_abs_strike != other.use_abs_strike:
            return False
        return self.use_ctp_config == other.use_ctp_config

    def __hash__(self):
        return hash((self.sim_path, self.calib_time_steps, self.pricing_time_steps, self.x_steps, self.model_name, self.exercise_probability_threshold,
                     self.random_number, self.mc_type, self.mc_discretization_scheme, self.antithetic, self.use_model_date, self.processing_unit,
                     self.seed, self.use_abs_strike, self.use_ctp_config))


if __name__ == '__main__':
    param = EQNxModelParams({"sim_path": 200})
    print(param.sim_path, param.calib_time_steps, param.pricing_time_steps)
