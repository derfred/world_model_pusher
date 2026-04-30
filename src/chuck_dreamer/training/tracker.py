class Tracker:
  def __init__(self, config, data={}, parent=None):
    self.config  = config
    self.data    = data
    self._parent = parent

  def init(self, data={}):
    self.data = data

    if self.config.logging.logger == "wandb":
      import wandb
      wandb.init(project=self.config.logging.project_name, config=self.config)
    elif self.config.logging.logger == "trackio":
      import trackio
      trackio.init(project=self.config.logging.project_name, config=self.config)
    else:
      self._tracker = None

  def log(self, data: dict, **kwargs):
    if self._parent:
      self._parent.log({**self.data, **data}, **kwargs)
    elif self.config.logging.logger == "wandb":
      import wandb
      wandb.log({**data, **kwargs})
    elif self.config.logging.logger == "trackio":
      import trackio
      trackio.log({**data, **kwargs})

  def derive(self, data: dict):
    return Tracker(self.config, data=data, parent=self)
