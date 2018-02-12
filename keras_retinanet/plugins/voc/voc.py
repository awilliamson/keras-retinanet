import keras_retinanet.utils.plugin as plugins
# See https://yapsy.readthedocs.io/en/latest/Advices.html#plugin-class-detection-caveat
# Caveat surrounding import. Must us 'as' rather than directly importing DatasetPlugin

class VocPlugin(plugins.DatasetPlugin):
    def __init__(self):
        super(VocPlugin, self).__init__()

        self.dataset_type = "pascal"

    def parse_args(self, subparsers):
        pascal_parser = subparsers.add_parser(self.dataset_type)
        pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

        return pascal_parser
