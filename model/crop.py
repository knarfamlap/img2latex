
class Crop(object):
    """Crop the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, formula = sample['image'], sample['formula']

        h, w = image.shape[:2]  # get height and width

        new_h, new_w = self.output_size

        # image = image[h: h + new_h, w: w + new_w]

        return {'image': image, 'formula': formula}
