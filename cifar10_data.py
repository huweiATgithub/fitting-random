"""
cifar-10 dataset, with support for random labels
"""
import numpy as np

import torch
import torchvision.datasets as datasets


class CIFAR10RandomLabels(datasets.CIFAR10):
  """CIFAR10 dataset, with support for randomly corrupt labels.
  Params
  ------
  corrupt_prob: float
    Default 0.0. The probability of a label being replaced with random label.
  random_pixel_prob: float
    Default 0.0. The probability of an image being replaced with Gaussian distributed random pixels
  shuffle_pixels: int 0/1/2
    Default 0. 0: no permutation of pixels; 1: a random permutation is applied to all images; 2: different random permutations are applied to images
  num_classes: int
    Default 10. The number of classes in the dataset.
  """

  def __init__(self, corrupt_prob=0.0, random_pixel_prob=0.0, shuffle_pixels=0, num_classes=10, **kwargs):
    super(CIFAR10RandomLabels, self).__init__(**kwargs)
    self.n_classes = num_classes
    if corrupt_prob > 0:
      self.corrupt_labels(corrupt_prob)
    if random_pixel_prob > 0:
      self.corrupt_pixels(random_pixel_prob)
    if shuffle_pixels == 1:
      self.shuffle(shuffle_pixels)
    if shuffle_pixels == 2:
      self.shuffle(shuffle_pixels)

  def corrupt_labels(self, corrupt_prob):
    labels = np.array(self.train_labels if self.train else self.test_labels)
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(self.n_classes, mask.sum())
    labels[mask] = rnd_labels
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = [int(x) for x in labels]

    if self.train:
      self.train_labels = labels
    else:
      self.test_labels = labels

  def corrupt_pixels(self, random_pixel_prob):
    if self.train:
      data = self.train_data
    else:
      data = self.test_data
    #     np.random.seed(12345)
    mean = [x for x in [125.3, 123.0, 113.9]]
    std = [x for x in [63.0, 62.1, 66.7]]
    corrimgs = []
    for img in data:
      gaussian = []
      for k in zip(mean, std):
        gaussian.append(np.random.normal(k[0], k[1], size=(32, 32)))
      gaussian = np.array(gaussian).transpose((1, 2, 0))  # (32,32,3) array
      cor_img = (1 - random_pixel_prob) * img + random_pixel_prob * gaussian
      corrimgs.append(cor_img.astype('uint8'))
    if self.train:
      self.train_data = np.array(corrimgs)
    else:
      self.test_data = np.array(corrimgs)

  def shuffle(self, shuffle_pixels):
    if self.train:
      data = self.train_data
    else:
      data = self.test_data
    shuff_imgs = []
    if shuffle_pixels == 1:  # same permutation for all
      np.random.seed(12345)  # fix the random permutation
      shuffle_pat = np.random.permutation(32 * 32)
      for img in data:
        img = img.transpose(2, 0, 1).reshape(3, -1)  # 3*1024 array
        shuffle_img = img[:, shuffle_pat]
        shuffle_img = shuffle_img.transpose(1, 0).reshape(32, 32, 3)
        shuff_imgs.append(shuffle_img)
    if shuffle_pixels == 2:  # different permutations
      for img in data:
        shuffle_pat = np.random.permutation(32 * 32)
        img = img.transpose(2, 0, 1).reshape(3, -1)  # 3*1024 array
        shuffle_img = img[:, shuffle_pat]
        shuffle_img = shuffle_img.transpose(1, 0).reshape(32, 32, 3)
        shuff_imgs.append(shuffle_img)
    if self.train:
      self.train_data = np.array(shuff_imgs)
    else:
      self.test_data = np.array(shuff_imgs)


