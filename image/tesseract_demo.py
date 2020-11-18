import pytesseract
from PIL import Image

"""
path: 이미지 명
이미지에 있는 문장을 읽어오기
"""
def image_to_string(path):
    im = Image.open(path)
    return pytesseract.image_to_string(im)


if __name__ == '__main__':
    code = image_to_string("test.png")
    print(code)