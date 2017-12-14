import keras

from ..preprocessing.coco import CocoGenerator
import numpy as np
from pycocotools.coco import COCO
import os

class CocoSubsetGenerator( CocoGenerator ):
    def __init__(self, data_dir, set_name, image_data_generator, fraction, *args, **kwargs):

        self.data_dir = data_dir
        self.set_name = set_name
        self.coco = COCO(os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json'))

        self.load_classes()
        self.image_ids = self._filter(self.coco.getImgIds(), fraction)

        print("Set Name: {}".format(set_name))

        super(CocoGenerator, self).__init__(image_data_generator, **kwargs)

    def _filter(self, image_ids, fraction):

        if not 0 <= fraction <= 1.0:
            raise ValueError("Fractional value to subset generator invalid: {}".format(fraction))

        if not len(image_ids) > 0:
            raise ValueError("Image IDs has length 0")

        # Initialise bins for each class.
        cls = [[] for _ in range(self.num_classes())]

        # Filter each image_id into a bin representing its class.
        # Uses sorting to ensure deterministic behaviour.
        for val in image_ids:
            # Get annotations
            anno = self._load_annotations(val)

            # For each ground truth within the image.
            # Associate the image id with that class, and bin it.
            for example in anno:
                label = int(example[4])
                cls[label].append(val)

        # For each bin, take the fraction from 0 -> fraction
        # As images can contribute to many classes, use unique so that we only process the image once.
        fractional_bins = [b[: int(len(b)*fraction)] for b in cls]

        # np.unique is not order preserving, therefore use this trick to maintain order.
        # https://gist.github.com/lrhache/36a9a5ea5fe7e1f121e3#file-list_unique_benchmark-py-L40
        array_unique = np.unique([value for sublist in fractional_bins for value in sublist], return_index=True)
        dstack = np.dstack(array_unique)
        dstack.dtype = np.dtype([('v', dstack.dtype), ('i', dstack.dtype)])
        dstack.sort(order='i', axis=1)
        return dstack.flatten()['v'].tolist()

    def _load_annotations(self, image_val):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=image_val, iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def preprocess_image_inv(self, x):
        # Inverse of below.
        # mostly identical to "https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py"
        # except for converting RGB -> BGR since we assume BGR already
        x = x.astype(keras.backend.floatx())
        if keras.backend.image_data_format() == 'channels_first':
            if x.ndim == 3:
                x[0, :, :] += 103.939
                x[1, :, :] += 116.779
                x[2, :, :] += 123.68
            else:
                x[:, 0, :, :] += 103.939
                x[:, 1, :, :] += 116.779
                x[:, 2, :, :] += 123.68
        else:
            x[..., 0] += 103.939
            x[..., 1] += 116.779
            x[..., 2] += 123.68

        return x