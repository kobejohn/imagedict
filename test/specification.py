import unittest as ut
from os import path

import cv2
import numpy as np

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

    def test_lookup_with_get_method_has_optional_default(self):
        d = ImageDict()
        obj1 = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        obj2 = cv2.imread(path.join(_this_path, 'data', 'different_object.png'))
        lookup = cv2.imread(path.join(_this_path, 'data', 'lookup.png'))
        some_value = 1
        d[obj1] = some_value
        #use the same image in a lookup
        self.assertEqual(d.get(lookup), some_value)
        self.assertEqual(d.get(obj2, None), None)

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


@ut.skip #todo: wasn't as easy as I thought so do this if needed
class User_Gets_A_Manual_Confirmation_Image_Together_With_Value(ut.TestCase):
    def test_user_gets_a_confirmation_image_with_value(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        lookup = cv2.imread(path.join(_this_path, 'data', 'lookup.png'))
        value_spec = 1
        d[obj] = value_spec
        value, confirmation_image = d.get(lookup, return_confirmation = True)
        self.assertEqual(value, value_spec)
        self.assertTrue(hasattr(confirmation_image, 'shape'), 'Looks like the confirmation image is not a cv numpy image.')
        self.assertTrue(hasattr(confirmation_image, 'dtype'), 'Looks like the confirmation image is not a cv numpy image.')


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
        keys_by_iterkeys = d.iterkeys()
        keys_by_keys = d.keys()
        self.assertItemsEqual(list(keys_by_iter), keys_specification)
        self.assertItemsEqual(list(keys_by_iterkeys), keys_specification)
        self.assertItemsEqual(keys_by_keys, keys_specification)


class ImageDict_Implements_Container_values(ut.TestCase):
    def test_ImageDict_provides_values_of_each_key_value_pair(self):
        d = ImageDict()
        obj1 = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        obj2 = cv2.imread(path.join(_this_path, 'data', 'different_object.png'))
        d[obj1] = 1
        d[obj2] = 2
        values_specification = (1, 2)
        values_by_itervalues = d.itervalues()
        values_by_values = d.values()
        self.assertItemsEqual(list(values_by_itervalues), values_specification)
        self.assertItemsEqual(values_by_values, values_specification)


class ImageDict_Implements_Container_iteritems_and_items(ut.TestCase):
    def test_ImageDict_provides_key_value_pairs(self):
        d = ImageDict()
        obj1 = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        mask1 = cv2.imread(path.join(_this_path, 'data', 'object_mask.png'))
        obj2 = cv2.imread(path.join(_this_path, 'data', 'different_object.png'))
        value_1 = 1
        value_2 = 2
        d[obj1, mask1] = value_1
        d[obj2] = value_2
        key1_spec = (obj1, mask1)
        key2_spec = (obj2, None)
        items_specification = ((key1_spec, value_1), (key2_spec, value_2))
        items_by_iteritems = d.iteritems()
        items_by_items = d.items()
        self.assertItemsEqual(list(items_by_iteritems), items_specification)
        self.assertItemsEqual(items_by_items, items_specification)


class ImageDict_Implements_Container_contains(ut.TestCase):
    def test_ImageDict_returns_true_if_key_exists(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        mask = cv2.imread(path.join(_this_path, 'data', 'object_mask.png'))
        d[obj, mask] = 1
        self.assertTrue(d.__contains__((obj, mask)))
        self.assertTrue((obj, mask) in d)
        self.assertTrue(d.has_key((obj, mask)))

    def test_ImageDict_returns_false_if_key_doesnt_exist(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        mask = cv2.imread(path.join(_this_path, 'data', 'object_mask.png'))
        obj2 = cv2.imread(path.join(_this_path, 'data', 'different_object.png'))
        d[obj, mask] = 1
        self.assertFalse(obj in d) #whole key is required, not only the image
        self.assertFalse(obj2 in d) #totally absent key


class ImageDict_Implements_Container_clear(ut.TestCase):
    def test_ImageDict_removes_all_items(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        d[obj] = 1
        d.clear()
        self.assertEqual(len(d), 0)


class ImageDict_Implements_Container_setdefault(ut.TestCase):
    def test_setdefault_with_an_existing_key_returns_the_existing_keys_value(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        original_value = 1
        new_value = 2
        d[obj] = original_value
        d.setdefault(obj, new_value)
        self.assertEqual(d._keypackages[0].value, original_value)

    def test_setdefault_with_a_key_that_doesnt_exist_sets_the_new_item_and_returns_the_new_value(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        new_value = 2
        d.setdefault(obj, new_value)
        self.assertEqual(d._keypackages[0].value, new_value)


class ImageDict_Implements_Container_pop_and_popitem(ut.TestCase):
    def test_pop_removes_the_provided_key_when_it_exists_and_returns_its_value(self):
        d = ImageDict()
        obj1 = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        value_1 = 1
        d[obj1] = value_1
        popped_value = d.pop(obj1)
        self.assertEqual(popped_value, value_1)
        self.assertEqual(len(d), 0) #should be empty after pop

    def test_pop_does_nothing_and_returns_the_provided_default_value_when_provided_key_doesnt_exist(self):
        d = ImageDict()
        obj1 = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        default_value = 1
        popped_value = d.pop(obj1, default_value) #pop a non-existent key
        self.assertEqual(popped_value, default_value)
        self.assertEqual(len(d), 0) #should still be empty

    def test_pop_raises_KeyError_when_provided_key_doesnt_exist_and_no_default_value_was_provided(self):
        d = ImageDict()
        obj1 = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        self.assertRaises(KeyError, d.pop, obj1)

    def test_popitem_removes_some_item_and_returns_it_as_a_key_value_pair(self):
        d = ImageDict()
        obj1 = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        value_1 = 1
        d[obj1] = value_1
        item_specification = ((obj1, None), value_1)
        popped_item = d.popitem()
        self.assertEqual(popped_item, item_specification)
        self.assertEqual(len(d), 0) #item should have been removed

    def test_popitem_raises_KeyError_when_ImageDict_is_empty(self):
        d = ImageDict()
        self.assertRaises(KeyError, d.popitem)


class ImageDict_Implements_Container_copy(ut.TestCase):
    def test_copy_is_a_new_object(self):
        d = ImageDict()
        d_copy = d.copy()
        self.assertIsNot(d_copy, d)

    def test_a_copy_has_same_keypackages_as_original(self):
        d = ImageDict()
        obj = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        value = 1
        d[obj] = value
        d_copy = d.copy()
        self.assertEqual(d._keypackages, d_copy._keypackages)


class ImageDict_Implements_Container_update(ut.TestCase):
    def test_arg_with_keys_sets_each_arg_item(self):
        obj1 = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        obj2 = cv2.imread(path.join(_this_path, 'data', 'different_object.png'))
        value_1 = 1
        value_2 = 2
        d_content = ImageDict()
        d_content[obj1] = value_1
        d_content[obj2] = value_2
        #create and try to update a blank ImageDict
        d = ImageDict()
        d.update(d_content)
        image_mask_values = ((kp.image, kp.mask, kp.value) for kp in d._keypackages)
        image_mask_values_spec = ((obj1, None, value_1), (obj2, None, value_2))
        self.assertItemsEqual(image_mask_values, image_mask_values_spec)

    def test_arg_without_keys_sets_sequence_of_key_value_pairs(self):
        obj1 = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        obj2 = cv2.imread(path.join(_this_path, 'data', 'different_object.png'))
        value_1 = 1
        value_2 = 2
        key_value_pairs = ((obj1, value_1), (obj2, value_2))
        #create and try to update a blank ImageDict
        d = ImageDict()
        d.update(key_value_pairs)
        items = [[kp.image, kp.mask, kp.value] for kp in d._keypackages]
        items_spec = ((obj1, None, value_1), (obj2, None, value_2))
        #below works like assertSequenceEqual but gets around numpy's demand for unambiguous equality operator
        #wish I could use ItemsEqual but I believe it has a problem even though it is supposed to have a non-hashable code path
        for (image_actual, mask_actual, value_actual), (image_spec, mask_spec, value_spec) in zip(items, items_spec):
            self.assertTrue(np.all(image_actual == image_spec))
            self.assertTrue(np.all(mask_actual == mask_spec))
            self.assertEqual(value_actual, value_spec)

    @ut.skip #read inside for hashing problem
    def test_kwarg_sets_as_with_an_arg_with_keys(self):
        obj1 = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        obj2 = cv2.imread(path.join(_this_path, 'data', 'different_object.png'))
        value_1 = 1
        value_2 = 2
        d_content = ImageDict()
        d_content[obj1] = value_1
        d_content[obj2] = value_2
        #create and try to update a blank ImageDict
        d = ImageDict()
        d.update(**d_content) #todo: this doesn't work. maybe can work around hash requirement?
        items = ((kp.image, kp.mask, kp.value) for kp in d._keypackages)
        items_spec = ((obj1, None, value_1), (obj2, None, value_2))
        #below works like assertSequenceEqual but gets around numpy's demand for unambiguous equality operator
        #wish I could use ItemsEqual but I believe it has a problem even though it is supposed to have a non-hashable code path
        for (image_actual, mask_actual, value_actual), (image_spec, mask_spec, value_spec) in zip(items, items_spec):
            self.assertTrue(np.all(image_actual == image_spec))
            self.assertTrue(np.all(mask_actual == mask_spec))
            self.assertEqual(value_actual, value_spec)


class ImageDict_Has_Basic_String_Representations(ut.TestCase):
    def setUp(self):
        self.obj1 = cv2.imread(path.join(_this_path, 'data', 'object.png'))
        self.mask1 = cv2.imread(path.join(_this_path, 'data', 'object_mask.png'))
        self.value1 = 1
        self.obj2 = cv2.imread(path.join(_this_path, 'data', 'different_object.png'))
        self.mask2 = None
        self.value2 = 2
        self.d = ImageDict()
        self.d[self.obj1, self.mask1] = self.value1
        self.d[self.obj2, self.mask2] = self.value2

    def test_unicode_shows_image_shape_for_each_key(self):
        shape1_spec = u'shape: ' + unicode(self.obj1.shape)
        shape2_spec = u'shape: ' + unicode(self.obj2.shape)
        self.assertEqual(unicode(self.d).count(shape1_spec), 1)
        self.assertEqual(unicode(self.d).count(shape2_spec), 1)

    def test_unicode_shows_count_of_masked_pixels_for_each_key(self):
        masked1_count = self.mask1.size -\
                        sum(1 for x in self.mask1.flat if x==0)
        masked2_count = 0
        masked1_spec = u'masked pixels: ' + unicode(masked1_count)
        masked2_spec = u'masked pixels: ' + unicode(masked2_count)
        self.assertEqual(unicode(self.d).count(masked1_spec), 1)
        self.assertEqual(unicode(self.d).count(masked2_spec), 1)

    def test_unicode_shows_count_of_keypoints_for_each_key(self):
        keypoints1_spec = u'keypoints: ' + unicode(len(self.d._keypackages[0].fingerprint.descriptors))
        keypoints2_spec = u'keypoints: ' + unicode(len(self.d._keypackages[1].fingerprint.descriptors))
        self.assertEqual(unicode(self.d).count(keypoints1_spec), 1)
        self.assertEqual(unicode(self.d).count(keypoints2_spec), 1)

    def test_unicode_shows_value_for_each_key(self):
        self.assertEqual(unicode(self.d).count(u'value: ' + unicode(self.value1)), 1)
        self.assertEqual(unicode(self.d).count(u'value: ' + unicode(self.value2)), 1)

    def test_str_is_simply_encoded_unicode(self):
        encoded_str_spec = unicode(self.d).encode('utf-8')
        self.assertEqual(str(self.d), encoded_str_spec)

    def test_repr_is_simply_unicode_with_class_name_header(self):
        repr_spec = u'{}\n{}'.format(u"<class 'imagedict.ImageDict'>", unicode(self.d))
        self.assertEqual(repr(self.d), repr_spec)




#todo: need init to allow input similar to update. maybe just calls update

