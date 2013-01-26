# -*- coding: utf-8 -*-
"""
imagedict
~~~~~~~~~

Emulates the basic behavior of a dict but with images as keys.

:copyright: (c) 2013 by John Nieri.
:license: MIT, see LICENSE for more details.
"""
from collections import namedtuple

import cv2
import numpy as np


class ImageDict(object):
    _KeyPackage = namedtuple("_KeyPackage", ('image', 'mask', 'fingerprint', 'value'))
    _Fingerprint = namedtuple("_Fingerprint", ('keypoints', 'descriptors'))

    def __init__(self):
        self._keypackages = list()
        self._detector = cv2.FastFeatureDetector()
        self._extractor = cv2.DescriptorExtractor_create('BRISK')
        self._matcher = cv2.BFMatcher(cv2.NORM_L2SQR)

    def get(self, key, default = None):
        raise NotImplementedError

    def __setitem__(self, key, value):
        """Set the given key and value.

        Allowed formats for the key are specified in the parse method
        Allowed formats for image and mask are specified in validate method

        Warning: providing a mask of None is valid, so make sure you actually
        have an image in the mask if you intend to use one.

        Note: a lot of validation is done to avoid confusing errors arising later

        """
        image, mask = self._parse_key_arg(key)
        self._validate_image_and_mask(image, mask)
        #if everything is ok, then create and store the package
        fingerprint = self._fingerprint(image, mask)
        keypackage = self._KeyPackage(image, mask, fingerprint, value)
        #todo: here is the place to search for presence of image (not mask) for overwriting
        self._keypackages.append(keypackage)

    def __getitem__(self, key):
        """Get the value for the given key or raise KeyError if not found.

        Allowed formats for the key are specified in the parse method
        Allowed formats for image and mask are specified in validate method

        Warning: providing a mask of None is valid, so make sure you actually
        have an image in the mask if you intend to use one.

        Note: a lot of validation is done to avoid confusing errors arising later

        """
        MINIMUM_GOOD_MATCHES = 3
        MINIMUM_MATCH_QUALITY = 0.1

        scene_image, scene_mask = self._parse_key_arg(key)
        self._validate_image_and_mask(scene_image, scene_mask)
        scene_keypoints, scene_descriptors = self._fingerprint(scene_image, scene_mask)
        if not self._keypackages:
            raise KeyError('No keys in this image dictionary.')
        #precalculate some scene properties
        scene_w, scene_h = scene_image.shape[0:2]
        scene_corners = np.float32([(0,0), (scene_w, 0), (scene_w, scene_h), (0, scene_h)])
        #for each key, decide if it has a good match in the scene
        matching_keypackages_with_quality = list()
        for kp in self._keypackages:
            obj_keypoints, obj_descriptors = kp.fingerprint
            matches = self._matcher.match(obj_descriptors, scene_descriptors)
            if len(matches) < MINIMUM_GOOD_MATCHES:
                continue
            #do some filtering of the matches to find the best ones
            distances = [match.distance for match in matches]
            min_dist = min(distances)
            avg_dist = sum(distances) / len(distances)
            min_dist = min_dist or avg_dist * 0.01 #if min_dist is zero, use a small percentage of avg instead
            good_matches = [match for match in matches if match.distance <= 3 * min_dist]
            #stop this keypackage if there are not enough matched points
            if len(good_matches) < MINIMUM_GOOD_MATCHES:
                continue
            #get the corresponding match positions
            obj_matched_points = np.array([obj_keypoints[match.queryIdx].pt for match in good_matches])
            scene_matched_points = np.array([scene_keypoints[match.trainIdx].pt for match in good_matches])
            #make a homography that describes how the key is oriented in the scene (rotated, scaled, etc.)
            homography, homography_inliers_mask = cv2.findHomography(obj_matched_points, scene_matched_points, cv2.RANSAC, 2.0) #2.0: should be very close
            inliers, total_good_matches = np.sum(homography_inliers_mask), len(homography_inliers_mask)
            #find the actual shape of the found key in the scene
            obj_w, obj_h = kp.image.shape[0:2]
            obj_corners = np.float32([(0,0), (obj_w, 0), (obj_w, obj_h), (0, obj_h)])
            found_obj_corners = cv2.perspectiveTransform(obj_corners.reshape(1, -1, 2), homography).reshape(-1, 2)
            #calculate the quality of the image match and keep/reject this keypackage
            quality = self._match_quality(obj_corners, found_obj_corners, inliers, total_good_matches)
            if quality < MINIMUM_MATCH_QUALITY:
                continue #finished with this key
            #store this keypackage and the quality of the match
            matching_keypackages_with_quality.append((kp, quality))
        #choose the best match from all qualifying matches
        try:
            best_kp, best_quality = max(matching_keypackages_with_quality, key=lambda x: x[1])
        except ValueError:
            raise KeyError('No good match found.')
        return best_kp.value

    def _fingerprint(self, image, mask):
        """Return the image descriptors and keypoints."""
        image_gray = self._get_grayscale(image)
        image_blurred = self._proportional_gaussian(image_gray)
        mask_gray = self._get_grayscale(mask) if mask is not None else None
        #create and return the fingerprint
        keypoints = self._detector.detect(image_blurred, mask_gray)
        keypoints, descriptors = self._extractor.compute(image_blurred, keypoints)
        return self._Fingerprint(keypoints, descriptors)

    def _get_grayscale(self, image):
        """Convert BGR to grayscale or do nothing if image is already grayscale."""
        try:
            channels = image.shape[2] #i.e. rows, columns, channels <----
        except IndexError:
            channels = 1
        if channels == 1:
            return image
        elif channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError('Found an image with unsupported number of channels: {}'.format(image))

    def _proportional_gaussian(self, image):
        """Help objects with differing sharpness look more similar to the feature detector etc."""
        kernel_w = int(2.0 * round((image.shape[1]*0.005+1)/2.0)-1)
        kernel_h = int(2.0 * round((image.shape[0]*0.005+1)/2.0)-1)
        return cv2.GaussianBlur(image, (kernel_w, kernel_h), 0) #blur to make features match at different resolutions

    def _match_quality(self, obj_corners, found_obj_corners, inliers, total_good_matches):
        """Calculate the quality [0,1] of the best if any match found for obj within scene.

        Quality is calculated as  (inliers / matches) but passes through some
        filters that can set the quality to zero.
        """
        #return zero quality if the found object is too big or small vs original
        MAX_OBJECT_SCALING = 2
        obj_area = self._polygon_area(obj_corners)
        obj_in_scene_area = self._polygon_area(found_obj_corners)
        min_area = float(obj_area) / MAX_OBJECT_SCALING**2
        max_area = float(obj_area) * MAX_OBJECT_SCALING**2
        if not (min_area < obj_in_scene_area < max_area):
            return 0
        #if it looks reasonable, then base quality on inliers vs outliers
        quality = float(inliers) / total_good_matches
        return quality

    def _polygon_area(self, vertices):
        """Calculate the area of the polygon described by the vertices.

        Crossed polygons (not a good thing for this application) count part as
        negative area so this works well as a filter of twisted homographies.

        Thanks to Darel Rex Finley: http://alienryderflex.com/polygon_area/
        """
        area = 0.0
        X = [float(vertex[0]) for vertex in vertices]
        Y = [float(vertex[1]) for vertex in vertices]
        j = len(vertices) - 1
        for i in range(len(vertices)):
            area += (X[j] + X[i]) * (Y[j] - Y[i])
            j = i
        return abs(area) / 2 #abs in case it's negative

    def _parse_key_arg(self, key):
        """Parse the key as either image or (image, mask). Return image, mask."""
        try:
            image, mask = key #this breaks if image is a 2-row, 1-column image...
        except ValueError:
            image, mask = key, None
        return image, mask

    def _validate_image_and_mask(self, image, mask):
        """Validate the image and mask.

        Standard Exceptions:
        - TypeError: type of image or mask is not valid
        - ValueError: shape of image and mask do not match
        """
        if not self._is_on_whitelist(image, allow_None = False):
            raise TypeError('The type of the image is not supported.')
        if not self._is_on_whitelist(mask, allow_None = True):
            raise TypeError('The type of the mask is not supported.')
        #validate the mask match with the image
        if mask is not None:
            if image.shape[0:2] != mask.shape[0:2]:
                raise ValueError('The row x column dimensions of the mask do not match the image.')

    def _is_on_whitelist(self, image, allow_None = False):
        """Return True if the image is a valid type. False if it is not."""
        test_None = lambda x: x is None
        ducktests = [self._ducktest_opencv_numpy]
        if allow_None:
            ducktests.append(test_None)
        return any(test(image) for test in ducktests)

    def _ducktest_opencv_numpy(self, image):
        """Do a very quick ducktest of image as an opencv numpy image."""
        if not hasattr(image, 'shape'): return False
        if not hasattr(image, 'dtype'): return False
        return True #accept the image if it wasn't failed


#boilerplate multiprocessing protection
if __name__ == '__main__':
    pass
