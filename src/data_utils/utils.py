from typing import Iterable
from PIL import Image

def adapt_bb(frame_height: int, frame_width: int, bb_height: int, bb_width: int, left: int, top: int, right: int,
             bottom: int) -> (
        int, int, int, int):
    x_ctr = (left + right) // 2
    y_ctr = (bottom + top) // 2
    new_top = max(y_ctr - bb_height // 2, 0)
    new_bottom = min(new_top + bb_height, frame_height)
    new_left = max(x_ctr - bb_width // 2, 0)
    new_right = min(new_left + bb_width, frame_width)
    return new_left, new_top, new_right, new_bottom

def extract_bb(frame: Image.Image, bb: Iterable, scale: str, size: int) -> Image.Image:
    """
    Extract a face from a frame according to the given bounding box and scale policy
    :param frame: Entire frame
    :param bb: Bounding box (left,top,right,bottom) in the reference system of the frame
    :param scale: "scale" to crop a square with size equal to the maximum between height and width of the face, then scale to size
                  "crop" to crop a fixed square around face center,
                  "tight" to crop face exactly at the bounding box with no scaling
    :param size: size of the face
    :return:
    """
    left, top, right, bottom = bb
    if scale == "scale":
        bb_width = int(right) - int(left)
        bb_height = int(bottom) - int(top)
        bb_to_desired_ratio = min(size / bb_height, size / bb_width) if (bb_width > 0 and bb_height > 0) else 1.
        bb_width = int(size / bb_to_desired_ratio)
        bb_height = int(size / bb_to_desired_ratio)
        left, top, right, bottom = adapt_bb(frame.height, frame.width, bb_height, bb_width, left, top, right,
                                            bottom)
        face = frame.crop((left, top, right, bottom)).resize((size, size), Image.BILINEAR)
    elif scale == "crop":
        # Find the center of the bounding box and cut an area around it of height x width
        left, top, right, bottom = adapt_bb(frame.height, frame.width, size, size, left, top, right,
                                            bottom)
        face = frame.crop((left, top, right, bottom))
    elif scale == "tight":
        left, top, right, bottom = adapt_bb(frame.height, frame.width, bottom - top, right - left, left, top, right,
                                            bottom)
        face = frame.crop((left, top, right, bottom))
    else:
        raise ValueError('Unknown scale value: {}'.format(scale))

    return face

def extract_bb(frame: Image.Image, bb: Iterable, scale: str, size: int) -> Image.Image:
    """
    Extract a face from a frame according to the given bounding box and scale policy
    :param frame: Entire frame
    :param bb: Bounding box (left,top,right,bottom) in the reference system of the frame
    :param scale: "scale" to crop a square with size equal to the maximum between height and width of the face, then scale to size
                  "crop" to crop a fixed square around face center,
                  "tight" to crop face exactly at the bounding box with no scaling
    :param size: size of the face
    :return:
    """
    left, top, right, bottom = bb
    if scale == "scale":
        bb_width = int(right) - int(left)
        bb_height = int(bottom) - int(top)
        bb_to_desired_ratio = min(size / bb_height, size / bb_width) if (bb_width > 0 and bb_height > 0) else 1.
        bb_width = int(size / bb_to_desired_ratio)
        bb_height = int(size / bb_to_desired_ratio)
        left, top, right, bottom = adapt_bb(frame.height, frame.width, bb_height, bb_width, left, top, right,
                                            bottom)
        face = frame.crop((left, top, right, bottom)).resize((size, size), Image.BILINEAR)
    elif scale == "crop":
        # Find the center of the bounding box and cut an area around it of height x width
        left, top, right, bottom = adapt_bb(frame.height, frame.width, size, size, left, top, right,
                                            bottom)
        face = frame.crop((left, top, right, bottom))
    elif scale == "tight":
        left, top, right, bottom = adapt_bb(frame.height, frame.width, bottom - top, right - left, left, top, right,
                                            bottom)
        face = frame.crop((left, top, right, bottom))
    else:
        raise ValueError('Unknown scale value: {}'.format(scale))

    return face