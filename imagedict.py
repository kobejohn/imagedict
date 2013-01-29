# -*- coding: utf-8 -*-
"""
imagedict
~~~~~~~~~

Emulates the basic behavior of a dict but with images as keys.

:copyright: (c) 2013 by John Nieri.
:license: MIT, see LICENSE for more details.
"""
from collections import namedtuple

try:
    import cv2
except ImportError:
    message = 'Please install opencv >= 2.4.3 manually from http://sourceforge.net/projects/opencvlibrary/'
    message += '\nThen add the opencv/build/Python/2.x path to a .pth file in your site_packages directory.'
    message += '\nIn the meantime, trying to use the fallback cv2.pyd included with the distribution.'
    print message
    from _opencv_fallback import cv2
try:
    import numpy as np
except ImportError:
    message = 'It looks like setup was not able to compile and/or install numpy.'
    message += '\nPlease install numpy. See this if you have problems on windows: http://stackoverflow.com/a/6753898/377366'
    raise ImportError(message)


class ImageDict(object):
    _KeyPackage = namedtuple("_KeyPackage", ('image', 'mask', 'fingerprint', 'value'))
    _Fingerprint = namedtuple("_Fingerprint", ('keypoints', 'descriptors'))

    def __init__(self):
        self._keypackages = list()
        self._detector = cv2.FastFeatureDetector()
        self._extractor = cv2.DescriptorExtractor_create('BRISK')
        self._matcher = cv2.BFMatcher(cv2.NORM_L2SQR)

    def __len__(self):
        return len(self._keypackages)

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
        new_keypackage = self._KeyPackage(image, mask, fingerprint, value)
        #overwrite the key if it exists
        try:
            i = self._index_of_key_in_keypackages(image, mask)
            self._keypackages[i] = new_keypackage
            return #finished if a key is overwritten
        except KeyError:
            self._keypackages.append(new_keypackage)

    def get(self, key, d = None, return_confirmation = False):
        """Get the value for the given key or provide the default value.

        return_confirmation: True/False to indicated if a special confirmation
            image should be returned with the value as (value, image)
        """
        #special confirmation path
        if return_confirmation:
            raise NotImplementedError #todo: maybe later if it's needed
        #standard get path
        try:
            return self.__getitem__(key)
        except KeyError:
            return d

    def setdefault(self, key, d):
        """D.setdefault(k,d) -> D.get(k,d), also set D[k]=d if k not in D

        Due to the keys being images, the default is required as there is no
        standard meaning for a default image.
        """
        try:
            value = self.__getitem__(key)
        except KeyError:
            self.__setitem__(key, d)
            value = d
        return value


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

    def __delitem__(self, key, index = None):
        """Delete the provided key. Raise KeyError if it doesn't exist."""
        #internal shortcut if index known:
        if index is not None:
            del(self._keypackages[index])
            return
        #normal path
        image, mask = self._parse_key_arg(key)
        self._validate_image_and_mask(image, mask)
        #find and delete the key
        try:
            i = self._index_of_key_in_keypackages(image, mask)
            del(self._keypackages[i])
            return
        except KeyError:
            raise KeyError('The provided key was not found.')

    def __iter__(self):
        """Provide an iterator over the keys."""
        for kp in self._keypackages:
            yield (kp.image, kp.mask)

    def iterkeys(self):
        return self.__iter__()

    def keys(self):
        return list(self.__iter__())

    def itervalues(self):
        for kp in self._keypackages:
            yield kp.value

    def values(self):
        return list(self.itervalues())

    def iteritems(self):
        for kp in self._keypackages:
            yield ((kp.image, kp.mask), kp.value)

    def items(self):
        return list(self.iteritems())

    #todo: how does a standard dict differentiate between d=None and not providing d? method signature says d=None
    def pop(self, key, d = 987654321):
        """D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
        If key is not found, d is returned if given, otherwise KeyError is raised.

        """
        image, mask = self._parse_key_arg(key)
        self._validate_image_and_mask(image, mask)
        try:
            i = self._index_of_key_in_keypackages(image, mask)
            value = self._keypackages[i].value
            self.__delitem__(key, index = i)
            return value
        except KeyError:
            pass
        if d == 987654321:
            raise KeyError('Key not found in dictionary.')
        else:
            return d

    def popitem(self):
        """D.popitem() -> (k, v), remove and return some (key, value) pair as a
        2-tuple; but raise KeyError if D is empty.

        """
        index = 0
        try:
            kp = self._keypackages[index]
        except IndexError:
            raise KeyError('Tried to popitem but the dict is empty.')
        item = ((kp.image, kp.mask), kp.value)
        self.__delitem__(None, index)
        return item

    def __contains__(self, key):
        """Return True if the image, mask combination is present."""
        image, mask = self._parse_key_arg(key)
        self._validate_image_and_mask(image, mask)
        try:
            index = self._index_of_key_in_keypackages(image, mask)
            return True #if index exists, then contains is true
        except KeyError:
            return False

    def has_key(self, key):
        return self.__contains__(key)

    def clear(self):
        self._keypackages = list()

    def copy(self):
        """Return a new ImageDict with the same items."""
        d = self.__class__()
        d._keypackages = list(self._keypackages)
        return d

    def update(self, E, **F):
        """D.update(E, **F) -> None. Update D from dict/iterable E and F.
        If E has a .keys() method, does: for k in E: D[k] = E[k]
        If E lacks .keys() method, does: for (k, v) in E: D[k] = v
        In either case, this is followed by: for k in F: D[k] = F[k]

        Note: This basically only works with mappables that can index an image

        """
        if hasattr(E, 'keys'):
            for k in E:
                self[k] = E[k]
        else:
            for (k, v) in E:
                self[k] = v
        for k in F:
            self[k] = F[k]

    def __unicode__(self):
        indent = u'    '
        item_string_list = list()
        for kp in self._keypackages:
            line_list = list()
            line_list.append(indent + u'shape: ' + unicode(kp.image.shape))
            if kp.mask is None:
                masked_pixels_count = 0
            else:
                masked_pixels_count = kp.mask.size -\
                                      sum(1 for p in kp.mask.flat if p==0)
                try:
                    channels = kp.mask.shape[2]
                except IndexError:
                    channels = 1
                masked_pixels_count = masked_pixels_count / channels
            line_list.append(indent + u'approximate masked pixels: ' + unicode(masked_pixels_count))
            line_list.append(indent + u'keypoints: ' + unicode(len(kp.fingerprint.descriptors)))
            line_list.append(indent + u'value: ' + unicode(kp.value))
            item_string_list.append('\n'.join(line_list))
        content_string = '\n----------------\n'.join(item_string_list)
        return u'{{\n{}\n}}'.format(content_string)

    def __str__(self):
        return self.__unicode__().encode('utf-8')

    def __repr__(self):
        return u'{}\n{}'.format(unicode(self.__class__), self.__unicode__())

    def _index_of_key_in_keypackages(self, image, mask):
        """Find the index of the validated key or raise KeyError if not found."""
        for i, kp in enumerate(self._keypackages):
            same_image = np.all(image == kp.image)
            same_mask = np.all(mask == kp.mask)
            if  same_image and same_mask:
                return i
        raise KeyError('Key not found in dictionary.')

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
        except TypeError:
            raise TypeError('The index does not seem valid. Please use either a cv numpy image or an image, mask pair.')
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
