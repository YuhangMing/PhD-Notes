# Random notes in python

### Setting up on Windows and trouble-shooting
[Instruction](https://www.neonscience.org/resources/learning-hub/tutorials/setup-git-bash-python)

[Conda not found on Git Bash](https://stackoverflow.com/questions/54501167/anaconda-and-git-bash-in-windows-conda-command-not-found)

### Package system
[official docs](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/)
[guide](https://docs.python-guide.org/writing/structure/)

### `__getattr__` vs `__getattribute__`
[link](https://stackoverflow.com/questions/3278077/difference-between-getattr-and-getattribute)
`__getattr__` is only invoked if the attribute wasn't found the usual ways. It's good for implementing a fallback for missing attributes. `__getattribute__` is invoked before looking at the actual attributes on the object, and so can be tricky to implement correctly. You can end up in infinite recursions very easily. Though one good reason to use `__getattribute__` is that you can prevent access to attributes and make them more secure as shown in the following example.

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

### OpenCV Draw Funcions
```
cv2.error: OpenCV(4.5.3) :-1: error: (-5:Bad argument) in function 'circle'
> Overload resolution failed:
>  - Layout of the output array img is incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)
>  - Expected Ptr<cv::UMat> for argument 'img'
```
- Error encountered when calling `cv2.circle`, same error happened with `rectangle`, etc.
- The reason: The img array is not stored continuously in memory
- The solution: `img = np.ascontiguousarray(img, dtype=np.uint8)`

[source](https://blog.actorsfit.com/a?ID=01250-431ce4e1-5ba4-43e8-b41c-45a150d71223)
