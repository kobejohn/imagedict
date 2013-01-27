import unittest as ut
from os import path

import cv2

from imagedict import ImageDict


_this_path = path.abspath(path.split(__file__)[0])


class User_Sets_A_New_Item(ut.TestCase):
    def test_setting_a_new_image_creates_a_new_item(self):
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        d = ImageDict()
        d[obj] = 1 #pass if no exception raised

    def test_user_isolates_important_parts_of_an_image_with_a_mask(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        mask = cv2.imread(path.join(_this_path, 'data', 'object_mask.png'))
        d[obj, mask] = 1 #pass if no problems

    def test_setting_an_image_that_is_not_on_the_whitelist_of_types_causes_TypeError(self):
        d = ImageDict()
        invalid_obj = 1
        some_value = None
        self.assertRaises(TypeError, d.__setitem__, invalid_obj, some_value)

    def test_setting_a_mask_that_is_not_on_the_whitelist_of_types_causes_TypeError(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        invalid_mask = 1
        some_value = None
        self.assertRaises(TypeError, d.__setitem__, (obj, invalid_mask), some_value)

    def test_setting_a_mask_causes_ValueError_if_not_the_same_size_as_the_main_image(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        bad_mask = cv2.imread(path.join(_this_path, 'data', 'object_bad_mask.png'))
        some_value = None
        self.assertRaises(ValueError, d.__setitem__, (obj, bad_mask), some_value)


class User_Sets_An_Existing_Item(ut.TestCase):
    def test_setting_the_same_image_with_the_same_mask_overwrites_the_existing_item(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        mask = cv2.imread(path.join(_this_path, 'data', 'object_mask.png'))
        first_value = 1
        second_value = 2
        d[obj, mask] = first_value
        #set the same obj with a new mask
        d[obj, mask] = second_value
        #confirm that there is only one item
        self.assertEqual(len(d), 1)
        #confirm the value was updated
        self.assertEqual(d._keypackages[0].value, second_value)

    def test_setting_the_same_image_with_a_different_mask_creates_a_new_item(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        mask1 = cv2.imread(path.join(_this_path, 'data', 'object_mask.png'))
        mask2 = None
        first_value = 1
        second_value = 2
        d[obj, mask1] = first_value
        #set the same obj with a new mask
        d[obj, mask2] = second_value
        #confirm that there are now two items
        self.assertEqual(len(d), 2)


class User_Looks_Up_A_Value(ut.TestCase):
    def test_lookup_with_an_image_that_is_similar_to_one_of_the_keys_returns_that_keys_value(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        lookup = cv2.imread(path.join(_this_path, 'data', 'lookup.png'))
        some_value = 1
        d[obj] = some_value
        #use the same image in a lookup
        self.assertEqual(d[lookup], some_value)

    def test_user_isolates_important_parts_of_lookup_with_a_mask(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        lookup = cv2.imread(path.join(_this_path, 'data', 'lookup.png'))
        lookup_mask = cv2.imread(path.join(_this_path, 'data', 'lookup_mask.png'))
        d[obj] = 1
        x = d[lookup, lookup_mask] #pass if no problems

    def test_KeyError_for_lookup_with_an_image_that_matches_no_keys(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        different_lookup = cv2.imread(path.join(_this_path, 'data', 'different_object.png'))
        d[obj] = None
        self.assertRaises(KeyError, d.__getitem__, different_lookup)

    def test_using_a_lookup_image_that_is_not_on_the_whitelist_of_types_causes_TypeError(self):
        d = ImageDict()
        invalid_lookup = 1
        self.assertRaises(TypeError, d.__getitem__, invalid_lookup)

    def test_using_a_lookup_mask_that_is_not_on_the_whitelist_of_types_causes_TypeError(self):
        d = ImageDict()
        lookup = cv2.imread(path.join(_this_path, 'data', 'lookup.png'))
        invalid_mask = 1
        self.assertRaises(TypeError, d.__getitem__, (lookup, invalid_mask))

    def test_lookup_mask_causes_ValueError_if_not_the_same_size_as_the_lookup_image(self):
        d = ImageDict()
        lookup = cv2.imread(path.join(_this_path, 'data', 'lookup.png'))
        bad_mask = cv2.imread(path.join(_this_path, 'data', 'lookup_bad_mask.png'))
        self.assertRaises(ValueError, d.__getitem__, (lookup, bad_mask))


class User_Deletes_An_Item(ut.TestCase):
    def test_user_deletes_an_existing_item_by_providing_the_same_image_and_mask(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        mask = cv2.imread(path.join(_this_path, 'data', 'object_mask.png'))
        d[obj, mask] = 1
        #delete the existing object + mask combination
        del(d[obj, mask])
        #confirm that there are zero keys after deleting the only key
        self.assertEqual(len(d), 0)

    def test_user_gets_a_KeyError_when_deleting_a_non_existent_image_and_mask_combination(self):
        d = ImageDict()
        obj1 = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        mask = cv2.imread(path.join(_this_path, 'data', 'object_mask.png'))
        d[obj1, mask] = 1
        #confirm that deleting the same image with a different mask raises KeyError
        self.assertRaises(KeyError, d.__delitem__, (obj1, None))


class ImageDict_Has_Whitelist_Of_Allowed_Image_Types(ut.TestCase):
    def test_opencv_numpy_images_are_on_the_whitelist_of_types(self):
        color_image = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        d = ImageDict()
        self.assertTrue(d._is_on_whitelist(color_image))
        self.assertTrue(d._is_on_whitelist(gray_image))


class ImageDict_Implements_Container_len(ut.TestCase):
    def test_length_of_a_new_dict_is_zero(self):
        d = ImageDict()
        self.assertEqual(len(d), 0)

    def test_length_of_a_dict_with_x_keys_is_x(self):
        obj1 = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        obj2 = cv2.imread(path.join(_this_path, 'data', 'different_object.png'))
        d = ImageDict()
        d[obj1] = 1
        d[obj2] = 2
        self.assertEqual(len(d), 2)


class ImageDict_Implements_Container_iter_and_iterkeys(ut.TestCase):
    def test_iterating_over_an_ImageDict_provides_a_generator_of_keys_as_image_mask_pairs(self):
        d = ImageDict()
        obj1 = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        mask1 = cv2.imread(path.join(_this_path, 'data', 'object_mask.png'))
        obj2 = cv2.imread(path.join(_this_path, 'data', 'different_object.png'))
        d[obj1, mask1] = 1
        d[obj2] = 2
        keys_specification = [(obj1, mask1), (obj2, None)]
        keys_by_iter = (key for key in d)
        keys_by_iterkeys = (key for key in d.iterkeys())
        self.assertItemsEqual(list(keys_by_iter), keys_specification)
        self.assertItemsEqual(list(keys_by_iterkeys), keys_specification)


class ImageDict_Implements_Container_contains(ut.TestCase):
    def test_ImageDict_returns_true_if_key_exists(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        mask = cv2.imread(path.join(_this_path, 'data', 'object_mask.png'))
        d[obj, mask] = 1
        self.assertTrue(d.__contains__((obj, mask)))
        self.assertTrue((obj, mask) in d)

    def test_ImageDict_returns_false_if_key_doesnt_exist(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        mask = cv2.imread(path.join(_this_path, 'data', 'object_mask.png'))
        obj2 = cv2.imread(path.join(_this_path, 'data', 'different_object.png'))
        d[obj, mask] = 1
        self.assertFalse(obj in d) #whole key is required, not only the image
        self.assertFalse(obj2 in d) #totally absent key



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

#todo: specify a get_with_confirmation_image  (same as get but returns tuple of value and image)


#todo: override update
#D.update(E, **F) -> None. Update D from dict/iterable E and F.
#If E has a .keys() method, does: for k in E: D[k] = E[k]
#If E lacks .keys() method, does: for (k, v) in E: D[k] = v
#In either case, this is followed by: for k in F: D[k] = F[k]
