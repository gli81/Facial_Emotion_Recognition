# -*- coding: utf-8 -*-

# def mask(image, quadrant=None):
#     """
#     Transform the image tensor by setting designated quarters to 0's
#     @param image: the input image as a tensor
#     @param quadrant: a list designate which quadrant(s) to mask
#     """
#     if not quadrant: quadrant = []
#     transformed = image.clone()
#     _, len_, _ = transformed.shape
#     split_index = len_ // 2 ## we have square image
#     if 1 in quadrant:
#         transformed[:, split_index:, :split_index] = 0
#     if 2 in quadrant:
#         transformed[:, split_index:, split_index:] = 0
#     if 3 in quadrant:
#         transformed[:, :split_index, split_index:] = 0
#     if 4 in quadrant:
#         transformed[:, :split_index, :split_index] = 0
#     return transformed


class ImgMask:
    def __init__(self, quadrant=None):
        if not quadrant: self.quadrant = []
        else:
            self.quadrant = quadrant
    
    def __call__(self, img):
        transformed = img.clone()
        _, len_, _ = transformed.shape
        split_index = len_ // 2 ## we have square image
        if 1 in self.quadrant:
            transformed[:, :split_index:, split_index:] = 0
        if 2 in self.quadrant:
            transformed[:, :split_index, :split_index] = 0
        if 3 in self.quadrant:
            transformed[:, split_index:, :split_index] = 0
        if 4 in self.quadrant:
            transformed[:, split_index:, split_index:] = 0
        return transformed