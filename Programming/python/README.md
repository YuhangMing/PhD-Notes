# Random notes in python

### Package system
[official docs](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/)
[guide](https://docs.python-guide.org/writing/structure/)

### print with `sep`, `end` and `flush`
[link]https://realpython.com/lessons/sep-end-and-flush/

### PIL vs. OpenCV

PIL: Python Image Library Pillow, its for loading/processing/creating images
```python
from PIL import Image

size = (64, 64)
img = Image.open("test_image.jpg")
# img is PIL.Image class, it can be converted to numpy.ndarray class by np.array(img)

img.show()
img.thumbnail(size)
img.save(output_file, "JEPG")
```

OpenCV: open source Conputer Vision Library
```python
import cv2

img = cv2.imread("test_image.jpg", 0)
# NULL: BGR, 0: grayscale, -1: as it is
# img is numpy.ndarray class, colour images loaded as BGR by default
# use BGR2RGB for conversion

cv2.imshow("windowname", img)
cv2.waitkey(0)
cv2.destroyAllWindows()
cv2.imwrite("save.png", img)
```

NOTE, with the same image operations, OpenCV implementations are about **5x faster** than PIL implementations. see [link](https://www.kaggle.com/code/vfdev5/pil-vs-opencv/notebook) for detailed comparisons.

