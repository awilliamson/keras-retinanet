import keras_retinanet.utils.plugin as plugins
# See https://yapsy.readthedocs.io/en/latest/Advices.html#plugin-class-detection-caveat
# Caveat surrounding import. Must us 'as' rather than directly importing DatasetPlugin

class VocPlugin(plugins.DatasetPlugin):
    def __init__(self):
        super(VocPlugin, self).__init__()
