Quick Usage
===========
::

    #some extra imports just for loading an image
    from StringIO import StringIO
    import urllib2
    import cv2
    import numpy

    from imagedict import ImageDict

    #some images to use (url load by evanchin: http://stackoverflow.com/a/13329446/377366 )
    def cvimage_from_url(url, cv2_img_flag=0):
        request = urllib2.urlopen(url)
        img_array = numpy.asarray(bytearray(request.read()), dtype=numpy.uint8)
        return cv2.imdecode(img_array, cv2_img_flag)

    google_standard = cvimage_from_url('http://www.google.com/images/srpr/logo3w.png')
    bing_standard = cvimage_from_url('http://0.tqn.com/d/websearch/1/5/o/q/bing-logo.png')
    google_lookup = cvimage_from_url('http://www.brandingmagazine.com/wp-content/uploads/2011/11/Evolution-logo.jpg')

    #populate an ImageDict
    d = ImageDict()
    d[google_standard] = 'looks like google'
    d[bing_standard] = 'looks like bing'

    #look up another google logo
    d[google_lookup]
    #hopefully that identified this image as the google logo!


Introduction
============
ImageDict emulates the basic behavior of a dict but with images as keys.
The idea is to take a lookup image, find the most similar key and return the
value for that key. It can also reject an input image as too dissimilar and
raise a KeyError.

Both setting and looking up keys can be done with optional masks that isolate
the important parts of each image.

On the technical side, this does not use template matching. Instead it uses
a technique called feature detection and specifically a type of feature
detection that is able to deal with some level of transformation between
the lookup and the key (e.g. scale, rotation).


Usage Guidelines
================
Since comparisons of non-identical images involve an element of uncertainty,
there are a few things you can do to make this work well.

- Key Images: Tightly cropped
- Lookup Images: Loosely cropped
- All Images: Provide a mask over non-relevant sections if possible
- All Images: The key image and the object in the lookup image will match
  more easily if they are similar resolutions


Copyright
=========
Copyright (c) 2013 John Nieri and contributors under MIT License. See LICENSE
in this repository or distribution for details.