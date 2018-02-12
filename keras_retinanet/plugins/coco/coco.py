import keras_retinanet.utils.plugin as plugins
# See https://yapsy.readthedocs.io/en/latest/Advices.html#plugin-class-detection-caveat
# Caveat surrounding import. Must us 'as' rather than directly importing DatasetPlugin

class CocoPlugin(plugins.DatasetPlugin):
    def __init__(self):
        super(CocoPlugin, self).__init__()

    def parser_args(self, subparsers):
        coco_parser = subparsers.add_parser('coco')
        coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

        return coco_parser
