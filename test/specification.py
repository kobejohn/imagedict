import unittest as ut
from os import path

import cv2

from imagedict import ImageDict


_this_path = path.abspath(path.split(__file__)[0])


class User_Sets_A_New_Item(ut.TestCase):
    def test_key_for_a_new_item_is_an_image(self):
        key_image = cv2.imread(path.join(_this_path, 'data', 'key.png'))
        d = ImageDict()
        d[key_image] = 1 #pass if no exception raised

    def test_setting_keys_that_are_not_on_the_whitelist_of_types_causes_TypeError(self):
        d = ImageDict()
        invalid_key = 1
        some_value = None
        self.assertRaises(TypeError, d.__setitem__, invalid_key, some_value)

#todo:
@ut.skip
class User_Sets_An_Existing_Item(ut.TestCase):
    def test_XYZ(self):
        raise NotImplementedError


class User_Can_Include_A_Mask_When_Setting_An_Item(ut.TestCase):
    def test_user_isolates_important_parts_of_new_key_with_a_mask(self):
        d = ImageDict()
        image = cv2.imread(path.join(_this_path, 'data', 'key.png'))
        mask = cv2.imread(path.join(_this_path, 'data', 'key_mask.png'))
        d[image, mask] = 1 #pass if no problems

    def test_setting_masks_that_are_not_on_the_whitelist_of_types_cause_TypeError(self):
        d = ImageDict()
        image = cv2.imread(path.join(_this_path, 'data', 'key.png'))
        invalid_mask = 1
        some_value = None
        self.assertRaises(TypeError, d.__setitem__, (image, invalid_mask), some_value)

    def test_new_key_mask_causes_ValueError_if_not_the_same_size_as_the_key_image(self):
        d = ImageDict()
        image = cv2.imread(path.join(_this_path, 'data', 'key.png'))
        bad_mask = cv2.imread(path.join(_this_path, 'data', 'key_bad_mask.png'))
        some_value = None
        self.assertRaises(ValueError, d.__setitem__, (image, bad_mask), some_value)


class User_Looks_Up_A_Value(ut.TestCase):
    def test_lookup_with_an_image_that_is_similar_to_one_of_the_keys_returns_that_keys_value(self):
        d = ImageDict()
        key = cv2.imread(path.join(_this_path, 'data', 'key.png'))
        lookup = cv2.imread(path.join(_this_path, 'data', 'lookup.png'))
        some_value = 1
        d[key] = some_value
        #use the same image in a lookup
        self.assertEqual(d[lookup], some_value)

    def test_KeyError_for_lookup_with_an_image_that_matches_no_keys(self):
        d = ImageDict()
        key = cv2.imread(path.join(_this_path, 'data', 'key.png'))
        different_lookup = cv2.imread(path.join(_this_path, 'data', 'different_lookup.png'))
        d[key] = None
        self.assertRaises(KeyError, d.__getitem__, different_lookup)


class User_Can_Include_A_Mask_When_Looking_Up_A_Key(ut.TestCase):
    def test_user_isolates_important_parts_of_lookup_with_a_mask(self):
        d = ImageDict()
        key = cv2.imread(path.join(_this_path, 'data', 'key.png'))
        lookup = cv2.imread(path.join(_this_path, 'data', 'lookup.png'))
        lookup_mask = cv2.imread(path.join(_this_path, 'data', 'lookup_mask.png'))
        d[key] = 1
        x = d[lookup, lookup_mask] #pass if no problems

    def test_using_lookup_masks_that_are_not_on_the_whitelist_of_types_causes_TypeError(self):
        d = ImageDict()
        image = cv2.imread(path.join(_this_path, 'data', 'lookup.png'))
        invalid_mask = 1
        self.assertRaises(TypeError, d.__getitem__, (image, invalid_mask))

    def test_lookup_mask_causes_ValueError_if_not_the_same_size_as_the_lookup_image(self):
        d = ImageDict()
        image = cv2.imread(path.join(_this_path, 'data', 'lookup.png'))
        bad_mask = cv2.imread(path.join(_this_path, 'data', 'lookup_bad_mask.png'))
        self.assertRaises(ValueError, d.__getitem__, (image, bad_mask))


class ImageDict_Has_Whitelist_Of_Allowed_Image_Types(ut.TestCase):
    def test_opencv_numpy_images_are_on_the_whitelist_of_types(self):
        color_image = cv2.imread(path.join(_this_path, 'data', 'key.png'))
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        d = ImageDict()
        self.assertTrue(d._is_on_whitelist(color_image))
        self.assertTrue(d._is_on_whitelist(gray_image))



#todo: specify a get_with_confirmation_image  (same as get but returns tuple of value and image)






#__len__(	self)
#Called to implement the built-in function len(). Should return the length of the object, an integer >= 0. Also, an object that doesn't define a __nonzero__() method and whose __len__() method returns zero is considered to be false in a Boolean context.

#__delitem__(	self, key)
#Called to implement deletion of self[key]. Same note as for __getitem__(). This should only be implemented for mappings if the objects support removal of keys, or for sequences if elements can be removed from the sequence. The same exceptions should be raised for improper key values as for the __getitem__() method.

#__iter__(	self)
#This method is called when an iterator is required for a container. This method should return a new iterator object that can iterate over all the objects in the container. For mappings, it should iterate over the keys of the container, and should also be made available as the method iterkeys().
#Iterator objects also need to implement this method; they are required to return themselves. For more information on iterator objects, see ``Iterator Types'' in the Python Library Reference.

#The membership test operators (in and not in) are normally implemented as an iteration through a sequence. However, container objects can supply the following special method with a more efficient implementation, which also does not require the object be a sequence.
#__contains__(	self, item)
#Called to implement membership test operators. Should return true if item is in self, false otherwise. For mapping objects, this should consider the keys of the mapping rather than the values or the key-item pairs.


#It is also recommended that mappings provide the methods behaving similar to those for Python's standard dictionary objects.
# base these on the above parts, especial get set keys
# keys(),
# values(),
# items(),
# has_key(),
# get(),
# clear(),
# setdefault(),
# iterkeys(),
# itervalues(),
# iteritems(),
# pop(),
# popitem(),
# copy(),
# update()


#__unicode__
#__str__
#__repr__










##some extra imports just for loading an image
#from StringIO import StringIO
#import urllib
#import cv2
#numpy
#
#from imagedict import ImageDict
#
##some images to use (url load by evanchin: http://stackoverflow.com/a/13329446/377366 )
#def cvimage_from_url(url, cv2_img_flag=0):
#    request = urlopen(url)
#    img_array = np.asarray(bytearray(request.read()), dtype=np.uint8)
#    return cv2.imdecode(img_array, cv2_img_flag)
#google_standard = cvimage_from_url('https://www.google.com/images/srpr/logo3w.png')
#bing_standard = cvimage_from_url('http://0.tqn.com/d/websearch/1/5/o/q/bing-logo.png')
#google_lookup = cvimage_from_url('http://mexico.cnn.com/media/2012/11/22/google-buscador-archivo.jpg')
#
##populate an ImageDict
#imagedict = ImageDict()
#imagedict[google_standard] = 'that's an image of the google logo'
#imagedict[bing_standard] = 'that's an image of the bing logo'
#
##look up a scaled, partial, rotated (in 2 axes), out of focus google logo
#imagedict[google_lookup]
##hopefully that identified this image as the google logo!


if __name__ == '__main__':
    ut.main()
