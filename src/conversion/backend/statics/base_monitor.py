class BaseMonitor:
    def __init__(self):
        self.hooks = []
        self.monitored_layers = []
        self.records = []
        self.name_records_index = {}
        self._enable = True
    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        elif isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        else:
            raise ValueError(i)
    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.name_records_index.items():
            v.clear()
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    def __del__(self):
        self.remove_hooks()
