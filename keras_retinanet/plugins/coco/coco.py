import keras_retinanet.utils.plugin as plugins
# See https://yapsy.readthedocs.io/en/latest/Advices.html#plugin-class-detection-caveat
# Caveat surrounding import. Must us 'as' rather than directly importing DatasetPlugin

from keras_retinanet.preprocessing.coco import CocoGenerator


class CocoPlugin(plugins.DatasetPlugin):
    def __init__(self):
        super(CocoPlugin, self).__init__()

        self.dataset_type = "coco"

    def parser_args(self, subparsers):
        coco_parser = subparsers.add_parser(self.dataset_type)
        coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

        return coco_parser

    def get_generator(self, args, transform_generator=None):
        train_generator = CocoGenerator(
            args.coco_path,
            'train2017',
            transform_generator=transform_generator,
            batch_size=args.batch_size
        )

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            batch_size=args.batch_size
        )

        return {
            "train_generator": train_generator,
            "validation_generator": validation_generator
        }

