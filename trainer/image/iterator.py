import logging
import os

import numpy as np
from keras_preprocessing.image import Iterator, load_img, img_to_array, array_to_img
from keras_preprocessing.image.iterator import BatchFromFilesMixin
from keras_preprocessing.image.directory_iterator import DirectoryIterator

class MildIterator(Iterator):
    """Abstract base class for image data iterators.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self,
                 batch_size,
                 shuffle,
                 seed,
                 triplet_path):
        super(MildIterator, self).__init__(0, batch_size, shuffle, seed)
        count = 0
        f = open(triplet_path)
        f_read = f.read()

        q, p, n = set(), set(), set()
        for line in f_read.split('\n'):
            filenames = line.split(",")
            q.add(filenames[0])
            p.add(filenames[1])
            n.add(filenames[2])
            if len(line) > 1:
                count += 1

        logging.info(
            'Found %d images belonging to %d classes. Query Images: %d, Positive Image: %d, Negative Images: %d' % (
                count, 3, len(q), len(p), len(n)))

        count = count // batch_size * batch_size
        f.close()
        self.n = count * 3

        self.index_generator = self._flow_index()


class MildBatchFromFilesMixin(BatchFromFilesMixin):

    def _get_batches_of_transformed_samples(self, index_array):
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        filepaths = self.filepaths
        for i, j in enumerate(index_array):
            img = load_img(filepaths[j],
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            if hasattr(img, 'close'):
                img.close()
            if self.image_data_generator:
                params = self.image_data_generator.get_random_transform(x.shape)
                x = self.image_data_generator.apply_transform(x, params)
                x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(self.dtype)
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=self.dtype)
            # for i, label in enumerate(self.classes):
            #     batch_y[i, label] = 1.
        else:
            return batch_x
        # return [batch_x, batch_x, batch_x], batch_y
        return batch_x, batch_y
