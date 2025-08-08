class LocalBacktester:
    def __init__(self):
        pass

    def run(self, strategy, start_date, end_date, initial_state):
        strategy.preprocess()
        daily_states = strategy.evolve(start_date, end_date, initial_state)
        strategy.postprocess()
        return daily_states
