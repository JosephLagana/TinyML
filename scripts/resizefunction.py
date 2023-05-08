from PIL import Image
import numpy as np

img = Image.open('Try.jpg')
data = np.array(img)

img = img.resize((224,224))
img.save('Trynew.jpg')
