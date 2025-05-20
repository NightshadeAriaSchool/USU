#Requires Python 3.7
from __future__ import annotations
import pygame
import colorsys
import numpy as np
from abc import ABC, abstractmethod
import datetime

class _RectMode(ABC):
  @abstractmethod
  def corners(self, x1, y1, x2, y2):
    pass
  @abstractmethod
  def center(self, x1, y1, x2, y2):
    pass
  @abstractmethod
  def size(self, x1, y1, x2, y2):
    pass
  def radius(self, x1, y1, x2, y2):
    sx, sy = self.size(x1, y1, x2, y2)
    return sx * 0.5, sy * 0.5
class _RectModeCorners(_RectMode):
  _instance = None
  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  def corners(self, x1, y1, x2, y2):
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
  def center(self, x1, y1, x2, y2):
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5
  def size(self, x1, y1, x2, y2):
  	return np.abs(x1 - x2), np.abs(y1 - y2)
class _RectModeCorner(_RectMode):
  _instance = None
  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  def corners(self, x1, y1, x2, y2):
    x1, y1, x2, y2 = x1, y1, x1 + x2, y1 + y2
    return min((x1, x2)), min((y1, y2)), max((x1, x2)), max((y1, y2))
  def center(self, x1, y1, x2, y2):
    return x1 + x2 * 0.5, y1 + y2 * 0.5
  def size(self, x1, y1, x2, y2):
  	return np.abs(x2), np.abs(y2)
class _RectModeCenter(_RectMode):
  _instance = None
  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  def corners(self, x1, y1, x2, y2):
    x2, y2 = abs(x2) * 0.5, abs(y2) * 0.5
    return x1 - x2, y1 - y2, x1 + x2, y1 + y2
  def center(self, x1, y1, x2, y2):
    return x1, y1
  def size(self, x1, y1, x2, y2):
  	return np.abs(x2), np.abs(y2)
class _RectModeRadius(_RectMode):
  _instance = None
  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  def corners(self, x1, y1, x2, y2):
    x2, y2 = abs(x2), abs(y2)
    return x1 - x2, y1 - y2, x1 + x2, y1 + y2
  def center(self, x1, y1, x2, y2):
    return x1, y1
  def size(self, x1, y1, x2, y2):
  	return np.abs(x2) * 2, np.abs(y2) * 2

class _LineCap(ABC):
  @abstractmethod
  def points(self, p1, p2, size, detail):
    pass
class _LineCapRound(_LineCap):
  _instance = None
  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  def points(self, p1, p2, size, detail):
    size /= 2
    d = (p2[0] - p1[0], p2[1] - p1[1])
    d_len_inv = 1.0 / np.sqrt(d[0]*d[0]+ d[1]*d[1])
    d = (d[0] * d_len_inv * size, d[1] * d_len_inv * size)
    vec = (-d[1], d[0])
    angle_step = np.pi / detail
    points = []
    
    # Generate points for the first half of the rounded cap
    for i in range(detail + 1):
      angle = i * angle_step
      cos = np.cos(angle)
      sin = np.sin(angle)
      x = p1[0] + vec[0] * cos - vec[1] * sin  # Corrected to use p1[0] for x-coordinate
      y = p1[1] + vec[0] * sin + vec[1] * cos  # Corrected to use p1[1] for y-coordinate
      points.append((x, y))
    
    vec = (-vec[0], -vec[1])
    
    # Generate points for the second half of the rounded cap
    for i in range(detail + 1):
      angle = i * angle_step
      cos = np.cos(angle)
      sin = np.sin(angle)
      x = p2[0] + vec[0] * cos - vec[1] * sin  # Corrected to use p2[0] for x-coordinate
      y = p2[1] + vec[0] * sin + vec[1] * cos  # Corrected to use p2[1] for y-coordinate
      points.append((x, y))
    
    return points
class _LineCapSquare(_LineCap):
  _instance = None
  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  def points(self, p1, p2, size, detail):
    size /= 2
    d = (p2[0] - p1[0], p2[1] - p1[1])
    d_len_inv = 1.0 / np.sqrt(d[0]*d[0]+ d[1]*d[1])
    d = (d[0] * d_len_inv * size, d[1] * d_len_inv * size)
    vec = (-d[1], d[0])
    points = [
      (p2[0] + d[0] + vec[0], p2[1] + d[1] + vec[1]),
      (p2[0] + d[0] - vec[0], p2[1] + d[1] - vec[1]),
      (p1[0] - d[0] - vec[0], p1[1] - d[1] - vec[1]),
      (p1[0] - d[0] + vec[0], p1[1] - d[1] + vec[1]),
    ]
    return points
class _LineCapProject(_LineCap):
  _instance = None
  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  def points(self, p1, p2, size, detail):
    size /= 2
    d = (p2[0] - p1[0], p2[1] - p1[1])
    d_len_inv = 1.0 / np.sqrt(d[0]*d[0]+ d[1]*d[1])
    d = (d[0] * d_len_inv * size, d[1] * d_len_inv * size)
    vec = (-d[1], d[0])
    points = [
      (p2[0] + d[0] + vec[0], p2[1] + d[1] + vec[1]),
      (p2[0] + d[0] - vec[0], p2[1] + d[1] - vec[1]),
      (p1[0] - d[0] - vec[0], p1[1] - d[1] - vec[1]),
      (p1[0] - d[0] + vec[0], p1[1] - d[1] + vec[1]),
    ]
    return points

class _ShapeKind(ABC):
  @abstractmethod
  def finalize(self, vertices, closed):
    pass
class _ShapeKindPolygon:
  _instance = None
  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  def finalize(self, vertices, closed):
    polys, paths, points = [], [], []
    if len(vertices) == 0:
      pass
    elif len(vertices) == 1:
      points.append(vertices[0])
    elif len(vertices) == 2:
      paths.append([vertices[0], vertices[1]])
    else:
      poly, path = [], []
      for pt in vertices:
        poly.append(pt)
        path.append(pt)
      if closed:
        path.append(vertices[0])
      polys.append(poly)
      paths.append(path)
      return polys, paths, points
class _ShapeKindPoints:
  _instance = None
  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  def finalize(self, vertices, closed):
    polys, paths, points = [], [], []
    for pt in vertices:
      points.append(pt)
    return polys, paths, points
class _ShapeKindLines:
  _instance = None
  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  def finalize(self, vertices, closed):
    polys, paths, points = [], [], []
    for i in range(len(vertices) // 2):
      paths.append([vertices[i * 2], vertices[i * 2 + 1]])
    return polys, paths, points
class _ShapeKindTriangles:
  _instance = None
  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  def finalize(self, vertices, closed):
    polys, paths, points = [], [], []
    for i in range(len(vertices) // 3):
      paths.append([vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2], vertices[i * 3]])
      polys.append([vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]])
    return polys, paths, points
class _ShapeKindQuads:
  _instance = None
  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  def finalize(self, vertices, closed):
    polys, paths, points = [], [], []
    for i in range(len(vertices) // 4):
      paths.append([vertices[i * 4], vertices[i * 4 + 1], vertices[i * 4 + 2], vertices[i * 4 + 3], vertices[i * 4]])
      polys.append([vertices[i * 4], vertices[i * 4 + 1], vertices[i * 4 + 2], vertices[i * 4 + 3]])
    return polys, paths, points
class _ShapeKindTriangleFan:
  _instance = None
  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  def finalize(self, vertices, closed):
    polys, paths, points = [], [], []
    if len(vertices) >= 3:
      paths.append([vertices[0], vertices[1]])
    for i in range(2, len(vertices)):
      polys.append([vertices[0], vertices[i - 1], vertices[i]])
      paths.append([vertices[i - 1], vertices[i], vertices[0]])
    return polys, paths, points
class _ShapeKindTriangleStrip:
  _instance = None
  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  def finalize(self, vertices, closed):
    polys, paths, points = [], [], []
    if len(vertices) < 3:
      return polys, paths, points  # Not enough vertices to form a triangle

    # Add the first line segment
    paths.append([vertices[0], vertices[1]])

    # Iterate through the vertices to create triangles and line segments
    for i in range(2, len(vertices)):
      polys.append([vertices[i - 2], vertices[i - 1], vertices[i]])
      # Add line segment from current vertex to next if not at the end of the strip
      paths.append([vertices[i - 1], vertices[i], vertices[i - 2]])

    return polys, paths, points
class _ShapeKindQuadStrip:
  _instance = None
  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance
  def finalize(self, vertices, closed):
    polys, paths, points = [], [], []
    if len(vertices) > 3:
      paths.append([vertices[0], vertices[1]])
    for i in range(3, len(vertices), 2):
      polys.append([vertices[i - 2], vertices[i], vertices[i - 1], vertices[i - 3]])
      paths.append([vertices[i - 2], vertices[i], vertices[i - 1], vertices[i - 3]])
    return polys, paths, points


class Color:
  @staticmethod
  def byte(val):
    """Ensures the value is within the ubyte range."""
    return None if val is None else max(0, min(255, int(val)))
  
  @staticmethod
  def color(a, b=None, c=None, d=None, is_hsba:bool=False) -> tuple:
    """
    Returns an RGBA value based on the arguments.
    If is_hsba is True, interpret input as HSBA, otherwise as RGBA.
    Arguments can be 1 to 4 numeric values representing:
    (Gray), (Gray, Alpha), (Red, Green, Blue), or (Red, Green, Blue, Alpha).
    """
    
    if isinstance(a, tuple) and len(a) == 4:
      a, b, c, d = a[0], a[1], a[2], a[3]
    
    # Convert inputs to ubyte range
    a = Color.byte(a)
    b = Color.byte(b)
    c = Color.byte(c)
    d = Color.byte(d)
    
    # Handle HSBA color space
    if is_hsba:
      # Convert HSBA to RGBA
      if b is None:
        # Gray (brightness) with full saturation and opacity
        alpha = 255
        red, green, blue = Color.hsv_to_rgb(0, 255, a)
      elif c is None:
        # Gray (brightness) with full saturation and given opacity
        alpha = b
        red, green, blue = Color.hsv_to_rgb(0, 255, a)
      elif d is None:
        # Convert HSBA to RGBA
        alpha = 255
        red, green, blue = Color.hsv_to_rgb(a, b, c)
      else:
        # Convert HSBA to RGBA with given opacity
        alpha = d
        red, green, blue = Color.hsv_to_rgb(a, b, c)
      
      return (red, green, blue, alpha)
    
    # Handle RGBA color space
    else:
      if b is None:
        # Gray with full opacity
        return (a, a, a, 255)
      elif c is None:
        # Gray with given opacity
        return (a, a, a, b)
      elif d is None:
        # RGB with full opacity
        return (a, b, c, 255)
      else:
        # RGB with given opacity
        return (a, b, c, d)
  
  @staticmethod
  def hsv_to_rgb(h, s, v):
    h /= 255
    s /= 255
    v /= 255
    # Ensure hue is in the range [0, 1]
    h %= 1.0
    
    # Calculate chroma (colorfulness)
    c = v * s
    
    # Calculate the intermediate values
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c
    
    # Initialize RGB components
    r, g, b = 0, 0, 0
    
    if 0 <= h < 1/6:
        r, g, b = c, x, 0
    elif 1/6 <= h < 2/6:
        r, g, b = x, c, 0
    elif 2/6 <= h < 3/6:
        r, g, b = 0, c, x
    elif 3/6 <= h < 4/6:
        r, g, b = 0, x, c
    elif 4/6 <= h < 5/6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    
    # Adjust for brightness and add the base value
    r, g, b = int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)
    
    return r, g, b
  
  @staticmethod
  def alpha(color):
    return color[3]
  
  @staticmethod
  def red(color):
    return color[0]
  
  @staticmethod
  def green(color):
    return color[1]
  
  @staticmethod
  def blue(color):
    return color[2]
  
  @staticmethod
  def brightness(color):
    return max(color[0], color[1], color[2])
    
  @staticmethod
  def saturation(color):
    high = max(color[0], color[1], color[2])
    low = min(color[0], color[1], color[2])
    if high == 0: # Black
      return 0
    return low / high * 255
  
  @staticmethod
  def hue(color):
    high = max(color[0], color[1], color[2])
    low = min(color[0], color[1], color[2])
    
    rmax = color[0] == high
    gmax = color[1] == high
    bmax = color[2] == high
    
    if high == low: # No saturation
      return 0
    
    hue = 0
    if rmax:
      hue = (color[1] - color[2]) / (high - low)
    if gmax:
      hue = 2.0 + (color[2] - color[0]) / (high - low)
    if bmax:
      hue = 4.0 + (color[0] - color[1]) / (high - low)
    
    return ((hue + 6) % 6) * 255 / 6
  
  @staticmethod
  def darken(color, amount):
    if not isinstance(color, tuple) and not len(color) == 4:
      raise RuntimeError("The color is not a valid color.")
    amount = max(0, min(1, amount))
    
    return (color[0] * amount, color[1] * amount, color[2] * amount, color[3])
  
  @staticmethod
  def brighten(color, amount):
    if not isinstance(color, tuple) and not len(color) == 4:
      raise RuntimeError("The color is not a valid color.")
    amount = max(0, min(1, amount))
    
    brightness = max(color[0], color[1], color[2]) / 255
    influence = (1 - brightness) * amount
    new_color_rgb = (color[0] + color[0] * influence, color[1] + color[1] * influence, color[2] + color[2] * influence)
    return (new_color_rgb[0], new_color_rgb[1], new_color_rgb[2], color[3])
  
  
class _GraphicsConstants:
  CORNERS = _RectModeCorners()
  CORNER = _RectModeCorner()
  CENTER = _RectModeCenter()
  RADIUS = _RectModeRadius()
  
  ROUND = _LineCapRound()
  SQUARE = _LineCapSquare()
  PROJECT = _LineCapProject()
  
  POLYGON = _ShapeKindPolygon()
  POINTS = _ShapeKindPoints()
  LINES = _ShapeKindLines()
  TRIANGLES = _ShapeKindTriangles()
  QUADS = _ShapeKindQuads()
  TRIANGLE_FAN = _ShapeKindTriangleFan()
  TRIANGLE_STRIP = _ShapeKindTriangleStrip()
  QUAD_STRIP = _ShapeKindQuadStrip()

class Matrix:
  def __init__(self, m00, m01, m02, m10, m11, m12):
    self.m00 = m00
    self.m01 = m01
    self.m02 = m02
    self.m10 = m10
    self.m11 = m11
    self.m12 = m12

  @staticmethod
  def identity():
    return Matrix(1, 0, 0, 0, 1, 0)

  @staticmethod
  def from_ndarray(array):
    if array.shape == (3, 2):
      return Matrix(array[0, 0], array[1, 0], array[2, 0], array[0, 1], array[1, 1], array[1, 2])
    elif array.shape == (3, 3):
      return Matrix(array[0, 0], array[0, 1], array[0, 2],
                    array[1, 0], array[1, 1], array[1, 2])
    else:
      raise ValueError("Array must be 3x2 or 3x3")
  
  def transform(self, posx=0, posy=0, rot=0, sclx=1, scly=1):
    cos_theta = np.cos(rot)
    sin_theta = np.sin(rot)
    return Matrix(
      self.m00 * cos_theta * sclx - self.m01 * sin_theta * scly,
      self.m00 * sin_theta * sclx + self.m01 * cos_theta * scly,
      self.m00 * posx + self.m01 * posy + self.m02,
      self.m10 * cos_theta * sclx - self.m11 * sin_theta * scly,
      self.m10 * sin_theta * sclx + self.m11 * cos_theta * scly,
      self.m10 * posx + self.m11 * posy + self.m12
    )

  def translate(self, tx, ty):
    return self.transform(posx=tx, posy=ty)

  def rotate(self, angle):
    return self.transform(rot=angle)

  def scale(self, sx, sy):
    return self.transform(sclx=sx, scly=sy)

  def __matmul__(self, other):
    if isinstance(other, Matrix):
      return Matrix(
        self.m00 * other.m00 + self.m01 * other.m10,
        self.m00 * other.m01 + self.m01 * other.m11,
        self.m00 * other.m02 + self.m01 * other.m12 + self.m02,
        self.m10 * other.m00 + self.m11 * other.m10,
        self.m10 * other.m01 + self.m11 * other.m11,
        self.m10 * other.m02 + self.m11 * other.m12 + self.m12
      )
    elif isinstance(other, tuple) and len(other) == 2:
      x, y = other
      return (
        self.m00 * x + self.m01 * y + self.m02,
        self.m10 * x + self.m11 * y + self.m12
      )
    elif isinstance(other, list) and all(isinstance(item, tuple) and len(item) == 2 for item in other):
      return [(self.m00 * x + self.m01 * y + self.m02,
               self.m10 * x + self.m11 * y + self.m12) for x, y in other]
    else:
      raise ValueError("Operand must be a Matrix, a 2-tuple, or a list of 2-tuples, not " + str(type(other)))
  
  def invert(self):
    try:
      return self.invert_np()
    except np.linalg.LinAlgError:
      return self.invert_decomp()
  
  def invert_decomp(self):
    # Decompose the matrix
    tx, ty, r, sx, sy = self.decompose()

    # Reverse the components
    reversed_tx = -tx
    reversed_ty = -ty
    reversed_r = -r
    reversed_sx = 1 / sx
    reversed_sy = 1 / sy

    # Create a new Matrix instance with inverted values
    return Matrix.identity().translate(reversed_tx, reversed_ty).rotate(reversed_r).scale(reversed_sx, reversed_sy)
    
  def invert_np(self):
    # Create a 3x3 matrix from the class attributes
    matrix_array = np.array([[self.m00, self.m01, self.m02],
                             [self.m10, self.m11, self.m12],
                             [0, 0, 1]])
    
    # Compute the inverse using numpy
    inverse_matrix_array = np.linalg.inv(matrix_array)

    # Extract the inverted values
    inverted_m00, inverted_m01, inverted_m02 = inverse_matrix_array[0]
    inverted_m10, inverted_m11, inverted_m12 = inverse_matrix_array[1]

        # Create a new Matrix instance with inverted values
    inverted_matrix = Matrix(inverted_m00, inverted_m01, inverted_m02,
                             inverted_m10, inverted_m11, inverted_m12)

    return inverted_matrix

  def __str__(self):
    return f"Matrix({self.m00}, {self.m01}, {self.m02}, {self.m10}, {self.m11}, {self.m12})"
  
  def to_nparray3x3(self):
    raise NotImplementedError()
  
  def to_nparray4x4(self):
    raise NotImplementedError()
  
  def to_opengl(self):
    return (self.m00, self.m10, 0.0, 0.0,
            self.m01, self.m11, 0.0, 0.0,
            0.0,   0.0,   1.0, 0.0,
            self.m02, self.m12, 0.0, 1.0)
  
  def copy(self):
    return Matrix(self.m00, self.m01, self.m02, self.m10, self.m11, self.m12)
  
  def decompose(self):
    # Calculate the scale by considering the length of the matrix columns
    sx = np.sqrt(self.m00**2 + self.m10**2)
    sy = np.sqrt(self.m01**2 + self.m11**2)

    # Normalize the rotation part of the matrix
    norm = np.sqrt(self.m00**2 + self.m10**2)
    rot_m00 = self.m00 / norm
    rot_m10 = self.m10 / norm

    # Calculate the rotation angle in radians
    rotation = np.arctan2(rot_m10, rot_m00)

    # Normalize angle to be within -π to π
    rotation = (rotation + np.pi) % (2 * np.pi) - np.pi

    # Calculate the translation
    tx = self.m02
    ty = self.m12

    return tx, ty, - rotation, sx, sy

class Shape:
  def __init__(self, kind):
    self._vertices = None
    self._fills = None
    self._strokes = None
    self._weights = None
    self._fill = Color.color(0)
    self._stroke = Color.color(255)
    self._weight = 1
    self._children = []
    self._transform = Matrix.identity()
    self._closed = False
    self._kind = kind if kind is not None else _GraphicsConstants.POLYGON
    self._polys = None # TODO
    self._paths = None # TODO
    self._points = None # TODO
    
  def begin_shape(self):
    self._vertices = []
    self._fill = Color.color(0)
    self._stroke = Color.color(255)
    self._weight = 1
    self._closed = False
    self._polys = None
    self._paths = None
    self._points = None
    
  def end_shape(self, closed:bool=False):
    self._closed = closed
    self._polys, self._paths, self._points = self._kind.finalize(self._vertices, self._closed)
    
  def vertex(self, x, y):
    self._vertices.append((x, y))
  
  def fill(self, a1, a2=None, a3=None, a4=None):
    self._fill = Color.color(a1, a2, a3, a4)
  
  def stroke(self, a1, a2=None, a3=None, a4=None):
    self._stroke = Color.color(a1, a2, a3, a4)
  
  def stroke_weight(self, weight):
    self._weight = weight
  
  def translate(self, x, y):
    self._transform = self._transform.translate(x, y)
  
  def rotate(self, r):
    self._transform = self._transform.rotate(r)
  
  def scale(self, x, y):
    self._transform = self._transform.scale(x, y)
  
  def transform(self, posx=0, posy=0, rot=0, sclx=1, scly=1):
    self._transform = self._transform(posx, posy, rot, sclx, scly)
  
  def apply(self, matrix):
    self._transform = self._transform @ matrix
  
  def add_child(self, child):
    if not isinstance(child, Shape):
      raise TypeError("Child must be a Shape.")
    if any(arg == child for arg in self._children):
      return
    self._children.append(child)
  
  def remove_child(self, child):
    if not isinstance(child, (Shape, int)):
      raise TypeError("Child must be a Shape.")
    elif isinstance(child, Shape):
      if not any(arg == child for arg in self._children): # Might not be necessary
        return
      self._children.remove(child)
    elif isinstance(child, int):
      self._children.remove(self._children[child])
    
  def get_child(self, index:int):
    return self._children[index]
  
  def get_children(self):
    return [child for child in self._children]

class Graphics(pygame.Surface, _GraphicsConstants):
  _gl = None
  
  def __init__(self, arg1, arg2=None, use_gl=False):
    self._texture = None
    self._fbo = None

    if use_gl:
      if not self._import_opengl():
        use_gl = False
    self._uses_gl = use_gl
    
    if isinstance(arg1, Graphics):
      self._init_graphics(arg1)
    elif isinstance(arg1, np.ndarray):
      self._init_ndarray(arg1)
    elif isinstance(arg1, str):
      self._init_str(arg1)
    elif all(isinstance(arg, int) for arg in [arg1, arg2]):
      self._init_size(arg1, arg2)
    else:
      raise ValueError("Invalid arguments for Graphics constructor:" + str(type(arg1)) + ((" " + str(type(arg2))) if arg2 else ""))


    self.pixels = None
    
    self._do_fill = True
    self._do_stroke = True
    self._fill = Graphics.color(0, 0)
    self._stroke = Graphics.color(255)
    self._weight = 1
    
    self._rect_mode = Graphics.CORNER
    self._ellipse_mode = Graphics.CORNER
    self._ellipse_detail = 100
    self._stroke_cap = Graphics.PROJECT
    self._stroke_cap_detail = 10
    self._image_mode = Graphics.CORNER
    
    self._text_align_x = 0 # range 0 to 1, Left to Right
    self._text_align_y = 0 # range 0 to 1, Top to Bottom
    self._text_size = 20 # Text size
    self._text_leading = 1 # Space between lines
    self._text_font_name = Graphics.list_fonts()[1] # Font name. To be added.
    self._text_font = 10 # Font object. To be added.
    
    self._current_shape = None
    
    self._matrices = [Matrix.identity()]
  
  def _init_graphics(self, arg1:Graphics):
    self._init_size(arg1.width, arg1.height)
    self.load_pixels()
    arg1.load_pixels()
    for x in range(self.width):
      for y in range(self.height):
        self.pixels[x,y] = arg1.pixels[x,y]
    arg1.update_pixels()
    self.update_pixels()
    
  
  def _init_ndarray(self, arg1):
    # Check if the array has an alpha channel
    if arg1.shape[2] == 3:
      # Add an alpha channel with full opacity
      alpha_channel = np.full(arg1.shape[:2] + (1,), 255, dtype=np.uint8)
      arg1 = np.concatenate((arg1, alpha_channel), axis=2)

    self._init_gl(arg1.shape[0], arg1.shape[1])
    flag = pygame.SRCALPHA | (pygame.DOUBLEBUF | pygame.OPENGL if self._uses_gl else 0)

    super().__init__(arg1.shape[:2], flag)
    pygame.surfarray.blit_array(self, arg1)

    if self._uses_gl:
      # Bind the FBO
      Graphics._gl.glBindFramebuffer(Graphics._gl.GL_FRAMEBUFFER, self._fbo)

      # Bind the texture
      Graphics._gl.glBindTexture(Graphics._gl.GL_TEXTURE_2D, self._texture)

      # Render a quad with the texture applied
      Graphics._gl.glBegin(Graphics._gl.GL_QUADS)
      Graphics._gl.glTexCoord2f(0, 0); Graphics._gl.glVertex2f(-1, -1)
      Graphics._gl.glTexCoord2f(1, 0); Graphics._gl.glVertex2f(1, -1)
      Graphics._gl.glTexCoord2f(1, 1); Graphics._gl.glVertex2f(1, 1)
      Graphics._gl.glTexCoord2f(0, 1); Graphics._gl.glVertex2f(-1, 1)
      Graphics._gl.glEnd()

      # Unbind FBO and texture when done
      Graphics._gl.glBindTexture(Graphics._gl.GL_TEXTURE_2D, 0)
      Graphics._gl.glBindFramebuffer(Graphics._gl.GL_FRAMEBUFFER, 0)

  def _init_str(self, arg1):
    #Load image from the given path
    image_surface = pygame.image.load(arg1)
    
    # If the image does not have an alpha channel, convert it to include one
    if image_surface.get_alpha() is None:
      image_surface = image_surface.convert_alpha()
    
    self._init_gl(image_surface.get_width(), image_surface.get_height())
    flag = pygame.SRCALPHA | (pygame.DOUBLEBUF | pygame.OPENGL if self._uses_gl else 0)

    super().__init__(image_surface.get_size(), flag)
    # Blit the loaded image onto this surface
    self.blit(image_surface, (0, 0))

    if self._uses_gl:
      # Convert the pygame Surface to pixel data
      pixel_data = pygame.image.tostring(image_surface, "RGBA", True)

      # Bind the texture
      Graphics._gl.glBindTexture(Graphics._gl.GL_TEXTURE_2D, self._texture)

      # Upload pixel data to texture
      Graphics._gl.glTexImage2D(Graphics._gl.GL_TEXTURE_2D, 0, Graphics._gl.GL_RGBA, image_surface.get_width(), image_surface.get_height(), 0, Graphics._gl.GL_RGBA, Graphics._gl.GL_UNSIGNED_BYTE, pixel_data)

      # Set texture parameters (if not set already)
      Graphics._gl.glTexParameteri(Graphics._gl.GL_TEXTURE_2D, Graphics._gl.GL_TEXTURE_MIN_FILTER, Graphics._gl.GL_LINEAR)
      Graphics._gl.glTexParameteri(Graphics._gl.GL_TEXTURE_2D, Graphics._gl.GL_TEXTURE_MAG_FILTER, Graphics._gl.GL_LINEAR)

      # Bind the FBO
      Graphics._gl.glBindFramebuffer(Graphics._gl.GL_FRAMEBUFFER, self._fbo)

      # Render a quad with the texture applied (as shown in previous messages)

      # Unbind FBO and texture when done
      Graphics._gl.glBindTexture(Graphics._gl.GL_TEXTURE_2D, 0)
      Graphics._gl.glBindFramebuffer(Graphics._gl.GL_FRAMEBUFFER, 0)

  def _init_size(self, arg1, arg2):
    self._init_gl(arg1, arg2)
    flag = pygame.SRCALPHA | (pygame.DOUBLEBUF | pygame.OPENGL if self._uses_gl else 0)

    super().__init__((arg1, arg2), flag)

  def _init_gl(self, width, height):
    if self._uses_gl:
      # Generate and bind framebuffer
      self._fbo = Graphics._gl.glGenFramebuffers(1)
      Graphics._gl.glBindFramebuffer(Graphics._gl.GL_FRAMEBUFFER, self._fbo)
      
      # Generate and bind texture
      self._texture = Graphics._gl.glGenTextures(1)
      Graphics._gl.glBindTexture(Graphics._gl.GL_TEXTURE_2D, self._texture)
      Graphics._gl.glTexImage2D(Graphics._gl.GL_TEXTURE_2D, 0, Graphics._gl.GL_RGB, width, height, 0, Graphics._gl.GL_RGB, Graphics._gl.GL_UNSIGNED_BYTE, None)
      
      # Attach texture to FBO
      Graphics._gl.glFramebufferTexture2D(Graphics._gl.GL_FRAMEBUFFER, Graphics._gl.GL_COLOR_ATTACHMENT0, Graphics._gl.GL_TEXTURE_2D, self._texture, 0)

      # Check FBO status
      status = Graphics._gl.glCheckFramebufferStatus(Graphics._gl.GL_FRAMEBUFFER)
      if status != Graphics._gl.GL_FRAMEBUFFER_COMPLETE:
        self._uses_gl = False
        raise Exception(f"Framebuffer is not complete: Status {status}")

      # Unbind FBO and texture
      Graphics._gl.glBindFramebuffer(Graphics._gl.GL_FRAMEBUFFER, 0)
      Graphics._gl.glBindTexture(Graphics._gl.GL_TEXTURE_2D, 0)
    
  @property
  def width(self):
    """Width of the surface."""
    return self.get_width()

  @property
  def height(self):
    """Height of the surface."""
    return self.get_height()
  
  @property
  def _last_matrix(self):
    return self._matrices[len(self._matrices) - 1]
  
  @property
  def uses_gl(self):
    return self._uses_gl
  
  @_last_matrix.setter
  def _last_matrix(self, val):
    self._matrices[len(self._matrices) - 1] = val
  
  
  
  #------------------
  #Pixels
  #------------------
  def load_pixels(self):
    self.finalize()
    self.pixels = pygame.surfarray.pixels2d(self).copy()
    
  def update_pixels(self):
    if self.pixels is not None:
      buffer_surface = pygame.Surface(self.get_size())
      pygame.surfarray.blit_array(buffer_surface, self.pixels)
      self.blit(buffer_surface, (0, 0))
      
  def resize(self, w, h, to_native:bool = False):
    # Convert float dimensions to integers
    new_width = int(round(w))
    new_height = int(round(h))
    # Check for non-positive dimensions
    if new_width <= 0 or new_height <= 0:
      raise ValueError("Width and height must be positive numbers")
    # Use pygame.transform.scale to resize the surface
    resized_surface = pygame.transform.scale(self, (new_width, new_height))
    if to_native:
      return resized_surface
    # Create a new Graphics object with the resized surface
    new_graphics = Graphics(new_width, new_height)
    # Blit the resized surface onto the new Graphics object
    new_graphics.blit(resized_surface, (0, 0))
    return new_graphics
  
  def finalize(self):
    if not self.uses_gl:
      return
    
    Graphics._gl.glFinish()
    
    # Bind the FBO
    Graphics._gl.glBindFramebuffer(Graphics._gl.GL_FRAMEBUFFER, self._fbo)

    # Set the read buffer to the color attachment of the FBO
    Graphics._gl.glReadBuffer(Graphics._gl.GL_COLOR_ATTACHMENT0)

    # Read the pixels directly into the Surface
    buffer = Graphics._gl.glReadPixels(0, 0, self.width, self.height, Graphics._gl.GL_RGBA, Graphics._gl.GL_UNSIGNED_BYTE)

    # Unbind the FBO
    Graphics._gl.glBindFramebuffer(Graphics._gl.GL_FRAMEBUFFER, 0)

    # Write the buffer into self
    pygame.image.frombuffer(self, (self.width, self.height), buffer, "RGBA")
  
  #TODO: Dispose
  
  #------------------
  #draw
  #------------------
  def background(self, a1, a2=None, a3=None, a4=None, top_left=None, bottom_right=None, mode=None):
    color = self.color(a1, a2, a3, a4)  # Determine the color using the self.color method

    # If top_left is None or bottom_right is not None, set the mode to Graphics.CORNER
    if top_left is None or bottom_right is None:
      mode = Graphics.CORNER
      top_left = (0, 0)
      bottom_right = (self.width, self.height)
    if mode is None:
      mode = Graphics.CORNER

    # If both top_left and bottom_right are provided, calculate corners using the mode
    l, t = top_left
    r, b = bottom_right
    l, t, r, b = mode.corners(l, t, r, b)
    r -= l
    b -= t
    points = [(l, t), (l, b), (r, b), (r, t)]

    self._draw_shape_impl(points, color, None)
  
  def rect(self, x1, y1, x2, y2):
    if not all(isinstance(arg, (int, float, np.integer, np.floating)) for arg in [x1, y1, x2, y2]):
      raise TypeError("The arguments need to be numbers.")
    l, t, r, b = self._rect_mode.corners(x1, y1, x2, y2)
    points = [(l, t), (r, t), (r, b), (l, b)]
    self.polygon(points, closed=True)
  
  def square(self, x1, y1, s):
    self.rect(x1, y1, s, s)
    
  def ellipse(self, x1, y1, x2, y2):
    cx, cy = self._ellipse_mode.center(x1, y1, x2, y2)
    rx, ry = self._ellipse_mode.radius(x1, y1, x2, y2)
    points = []
    for i in range(self._ellipse_detail):
      angle = 2 * np.pi * i / self._ellipse_detail
      x = cx + rx * np.cos(angle)  # rx is a radius
      y = cy + ry * np.sin(angle)  # ry is a radius
      points.append((x, y))
    self.polygon(points, closed=True)
  
  def circle(self, x1, y1, s):
    self.ellipse(x1, y1, s, s)
  
  def arc(self, x1, y1, x2, y2, start_angle, end_angle, closed=False, pie=True):
    cx, cy = self._rect_mode.center(x1, y1, x2, y2)
    rx, ry = self._rect_mode.radius(x1, y1, x2, y2)
    fill = []
    stroke = []
    for i in range(self._ellipse_detail + 1):
      angle = start_angle + (end_angle - start_angle) * i / self._ellipse_detail
      x = cx + rx * np.cos(angle)  # rx is a radius
      y = cy + ry * np.sin(angle)  # ry is a radius
      fill.append((x, y))
      stroke.append((x, y))
    if pie:
      if closed:
        stroke.append((cx, cy))
      fill.append((cx, cy))
    
    self._draw_polygon_surface_impl(fill, self._fill, self._last_matrix)
    self._draw_line_path_impl(stroke, self._stroke, self._weight, self._stroke_cap_detail, self._stroke_cap, self._last_matrix, closed)
  
  def quad(self, x1, y1, x2, y2, x3, y3, x4, y4):
    self.polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)], closed=True)
  
  def triangle(self, x1, y1, x2, y2, x3, y3):
    self.polygon([(x1, y1), (x2, y2), (x3, y3)], closed=True)
  
  def polygon(self, points, closed=False):
    self._draw_polygon_surface_impl(points, self._fill, self._last_matrix)
    self._draw_line_path_impl(points, self._stroke, self._weight, self._stroke_cap_detail, self._stroke_cap, self._last_matrix, closed)
  
  def line(self, x1, y1, x2, y2):
    self.line_path(x1, y1, x2, y2)
    
  def line_path(self, *args):
    if len(args) % 2 != 0:
      raise ValueError("The number of arguments must be even (pairs of x, y coordinates).")
    
    points = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]
    self._draw_line_path_impl(points, self._stroke, self._weight, self._stroke_cap_detail, self._stroke_cap, self._last_matrix, closed=False)
  
  def bezier(self, detail, *args):
    if len(args) % 2 != 0:
      raise ValueError("The number of arguments must be even (pairs of x, y coordinates).")
    if detail < 2:
      raise ValueError("Detail must be at at least 2.")
    
    if len(args) == 0:
      return # Doesn't make sense
    if len(args) == 2:
      self.point(*args) # Don't unpack
    if len(args) == 4:
      self.line(*args) # Don't unpack
    
    points = [(args[i], args[i + 1]) for i in range(0, len(args), 2)]
    draw_points = []
    for i in range(detail):
      temp = [arg for arg in points]
      t = i / (detail - 1)
      while len(temp) > 1:
        for i in range(len(temp) - 1):
          a = temp[i]
          b = temp[i + 1]
          temp[i] = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        temp.pop()
      draw_points.append(temp[0])
    self.polygon(draw_points, closed=False) # Calling polygon instead of line_path, because it could be filled
  
  def point(self, x1, y1):
    self._draw_line_path_impl([(x1, y1)], self._stroke, self._weight, self._stroke_cap_detail, self._stroke_cap, self._last_matrix, closed=False)
  
  def image(self, graphics, x1, y1, x2=None, y2=None):
    if x2 is None or y2 == None:
      x2, y2 = graphics.width, graphics.height
    self._image_impl(graphics, x1, y1, x2, y2, self._image_mode)
  
  def shape(self, shape, x=None, y=None):
    if shape is None:
      return
    if not isinstance(shape, Shape):
      raise TypeError("Shape must be a Shape.")
    
    if x is None or y is None:
      x, y = 0, 0
    
    polys, paths, points = shape._polys, shape._paths, shape._points
    
    g_fill = self._fill
    g_stroke = self._stroke
    g_weight = self._weight
    g_do_stroke = self._do_stroke
    g_do_fill = self._do_fill
    
    self.push_matrix()
    self.apply(shape._transform)
    
    if shape._polys is not None and shape._paths is not None and shape._points is not None:
      self.no_stroke()
      self.fill(shape._fill)
      for poly in polys:
        self.polygon(poly, False)
      
      self.stroke(shape._stroke)
      self.stroke_weight(shape._weight)
      self.no_fill()
      for path in paths:
        self.polygon(path, False)
      for point in points:
        self.point(point[0], point[1])
    
    self._fill = g_fill  
    self._stroke = g_stroke
    self._weight = g_weight
    self._do_stroke = g_do_stroke
    self._do_fill = g_do_fill
    
    for s in shape._children:
      self.shape(s, x, y)
    
    self.pop_matrix()
  
  #TODO: effect
  #TODO: Shape class
  
  
  
  #------------------
  #shape
  #------------------
  def begin_shape(self, kind=_GraphicsConstants.POLYGON): #TODO: kind
    self._current_shape = Shape(kind)
    self._current_shape.begin_shape()
  
  def vertex(self, x, y):
    if self._current_shape is None:
      return
    self._current_shape.vertex(x, y)  # Add the vertex to the current shape
  
  def end_shape(self, closed=False):
    # Use the polygon method to draw the shape
    if self._current_shape is None:
      return
    self._current_shape.end_shape(closed)
    self._current_shape.fill(self._fill if self._do_fill else Color.color(0, 0, 0, 0))
    self._current_shape.stroke(self._stroke if self._do_stroke else Color.color(0, 0, 0, 0))
    self._current_shape.stroke_weight(self._weight)
    self.shape(self._current_shape, 0, 0)
    self._current_shape = None  # Clear the current shape
  
  
  
  
  #------------------
  #text
  #------------------
  def text_font(self, name=None):
    if name is None:
      return self._text_font
    """Set the font by name."""
    self._text_font_name = name
    self._text_font = pygame.font.Font(pygame.font.match_font(name), int(self._text_size))

  def text_size(self, size=None):
    if size is None:
      return self._text_size
    """Set the font size."""
    self._text_size = size
    if self._text_font_name:
      self._text_font = pygame.font.Font(pygame.font.match_font(self._text_font_name), int(size))
  
  def text_align(self, x, y):
    """Set the horizontal and vertical alignment of text."""
    self._text_align_x = x
    self._text_align_y = y

  def text_leading(self, leading):
    """Set the space between lines of text."""
    self._text_leading = leading
  
  @staticmethod
  def list_fonts():
    """List all available fonts."""
    return pygame.font.get_fonts()
  
  def text(self, obj, x, y):
    """Convert an object to text and render it onto a surface."""
    text_str = str(obj)  # Convert the object to a string
    lines = text_str.split("\n")
    
    # Get the width and height of the text
    
    text_surfaces = []
    text_width, text_height = 0, 0
    
    for i in range(len(lines)):
      if len(lines[0]) == 0:
        continue
      text_surface = self._text_font.render(lines[i], True, self._fill)
      text_surfaces.append(text_surface)
      text_width, text_height = max(text_width, text_surface.get_width()), max(text_height, text_surface.get_height())
    
    surface = pygame.Surface((int(text_width), int((text_height + self._text_leading) * len(lines))), pygame.SRCALPHA)
    for i in range(len(text_surfaces)):
      if not text_surfaces[i]:
        continue
      w, h = text_surfaces[i].get_width(), text_surfaces[i].get_height()
      draw_x = int((text_width - w) * self._text_align_x)
      draw_y = int((text_height + self._text_leading) * i)
      surface.blit(text_surfaces[i], (draw_x, draw_y))
    
    
    rx = x - surface.get_width() * self._text_align_x
    ry = y - surface.get_height() * self._text_align_y
    self._image_impl(surface, rx, ry, surface.get_width(), surface.get_height(), Graphics.CORNER)
  
  
  
  #------------------
  #colors
  #------------------
  def no_fill(self):
    self._do_fill = False
    
  def fill(self, a, b=None, c=None, d=None, is_hsba:bool=False):
    self._do_fill = True
    self._fill = Color.color(a, b, c, d, is_hsba)
    
  def no_stroke(self):
    self._do_stroke = False
    
  def stroke(self, a, b=None, c=None, d=None, is_hsba:bool=False):
    self._do_stroke = True
    self._stroke = Color.color(a, b, c, d, is_hsba)
    
  def stroke_weight(self, weight):
    if not isinstance(weight, (int, float, np.integer, np.floating)):
      raise TypeError("Weight must be a number.")
    self._weight = abs(float(weight))
  
  
  
  #------------------
  #modes
  #------------------
  def rect_mode(self, mode):
    if not isinstance(mode, _RectMode):
      raise TypeError("Mode must be a _RectMode. Use Graphics.CORNERS, Graphics.CORNER, Graphics.CENTER or Graphics.RADIUS.")
    self._rect_mode = mode
  
  def ellipse_mode(self, mode):
    if not isinstance(mode, _RectMode):
      raise TypeError("Mode must be a _RectMode. Use Graphics.CORNERS, Graphics.CORNER, Graphics.CENTER or Graphics.RADIUS.")
    self._ellipse_mode = mode
  
  def image_mode(self, mode):
    if not isinstance(mode, _RectMode):
      raise TypeError("Mode must be a _RectMode. Use Graphics.CORNERS, Graphics.CORNER, Graphics.CENTER or Graphics.RADIUS.")
    self._image_mode = mode
  
  def stroke_cap(self, mode):
    if not isinstance(mode, _LineCap):
      raise TypeError("Mode must be a _LineCap. Use Graphics.PROJECT, Graphics.SQUARE or Graphics.ROUND.")
    self._stroke_cap = mode
  
  def stroke_cap_detail(self, detail):
    if not isinstance(detail, (int, np.integer)):
      raise TypeError("Detail must be an integer.")
    if detail < 2:
      raise ValueError("Detail must be at least 2.")
    self._stroke_cap_detail = detail
  
  def ellipse_detail(self, detail):
    if not isinstance(detail, (int, np.integer)):
      raise TypeError("Detail must be an integer.")
    if detail < 4:
      raise ValueError("Detail must be at least 4.")
    self._ellipse_detail = detail
  
  
  
  #------------------
  #matrix
  #------------------
  def translate(self, x, y):
    self._last_matrix = self._last_matrix.translate(x, y)
  
  def rotate(self, r):
    self._last_matrix = self._last_matrix.rotate(r)
  
  def scale(self, x, y):
    self._last_matrix = self._last_matrix.scale(x, y)
  
  def apply(self, m):
    self._last_matrix = self._last_matrix @ m
  
  def push_matrix(self):
    self._matrices.append(self._last_matrix.copy())
  
  def pop_matrix(self):
    if len(self._matrices) == 1:
      raise RuntimeWarning("Can't pop matrix, because there is only one left.")
    self._matrices.pop()
  
  def reset_matrix(self):
    if len(self._matrices) == 1:
      self._last_matrix = Matrix.identity()
    else:
      self.pop_matrix()
      self.push_matrix()
  
  
  
  #------------------
  #private
  #------------------
  @staticmethod
  def _ensure_2d_int(point):
    """
    Ensures that a point is 2D and converts it to integers.
    
    :param point: A tuple representing a point, which can have float coordinates and be 3D.
    :return: A 2D point with integer coordinates.
    """
    if np.isnan(point[0]):
      point = (0, point[1])
    if np.isnan(point[1]):
      point = (point[0], 0)
    return (int(point[0]), int(point[1]))
  @staticmethod
  def _import_opengl():
    if Graphics._gl is not None:
      return True  # OpenGL is already imported
    try:
      from OpenGL import GL as _gl
      Graphics._gl = _gl
      del(_gl)
      return True
    except ImportError:
      return False
      
  def _draw_shape_pygame_impl(self, points, fill_color, matrix:Matrix = None):
    """
    Draws a filled polygon with the given points and fill color.

    :param points: A list of tuples representing the vertices of the polygon.
    :param fill_color: The color to fill the polygon with.
    """
    
    if len(points) < 2:
      return
    
    if matrix is not None:
      points = matrix @ points
    
    int_points = [Graphics._ensure_2d_int(point) for point in points]
    
    if fill_color[3] == 255:
      # Draw the polygon directly onto 'self'
      pygame.draw.polygon(self, fill_color, int_points)
    else:
      # Calculate the bounding box of the polygon
      min_x = min([p[0] for p in int_points])
      max_x = max([p[0] for p in int_points])
      min_y = min([p[1] for p in int_points])
      max_y = max([p[1] for p in int_points])
            
      # Create a surface for drawing the polygon with the size of the bounding box
      polygon_surface = pygame.Surface((max_x - min_x, max_y - min_y), pygame.SRCALPHA)
            
      # Offset the points by the top-left corner of the bounding box
      offset_points = [(p[0] - min_x, p[1] - min_y) for p in int_points]
            
      # Draw the polygon on the surface
      pygame.draw.polygon(polygon_surface, fill_color, offset_points)
            
      # Blit the polygon surface onto 'self' at the position of the bounding box
      self.blit(polygon_surface, (min_x, min_y))
  def _draw_shape_opengl_impl(self, points, fill_color, matrix:Matrix = None):
    """
    Draws a filled polygon with the given points and fill color using OpenGL.

    :param points: A list of tuples representing the vertices of the polygon.
    :param fill_color: The color to fill the polygon with.
    """
    
    if len(points) < 3:
      return  # Need at least 3 points to form a polygon

    # Convert fill_color to OpenGL color format
    gl_color = [c / 255 for c in fill_color]

    # Use Graphics._gl for OpenGL context
    Graphics._gl.glBegin(Graphics._gl.GL_POLYGON)
    Graphics._gl.glColor4fv(gl_color)
    
    if matrix is not None:
      # Load the matrix for transformation
      Graphics._gl.glPushMatrix()
      Graphics._gl.glMultMatrixf(matrix.to_opengl())

    for point in points:
      # Convert point to 3D by adding a z-coordinate of 0
      Graphics._gl.glVertex3f(point[0], point[1], 0)

    if matrix is not None:
      # Restore the previous matrix state
      Graphics._gl.glPopMatrix()

    Graphics._gl.glEnd()

  def _draw_shape_impl(self, points, fill_color, matrix:Matrix = None):
    if self._uses_gl:
      self._draw_shape_opengl_impl(points, fill_color, matrix)
    else:
      self._draw_shape_pygame_impl(points, fill_color, matrix)
  def _draw_polygon_surface_impl(self, points, fill_color, matrix:Matrix = None):
    """
    Draws a filled polygon on the surface if filling is enabled.

    This method delegates to _draw_shape_impl to perform the actual drawing.

    :param points: A list of tuples representing the vertices of the polygon.
    :param fill_color: The color to fill the polygon with.
    """
    if not self._do_fill:
      return
    
    self._draw_shape_impl(points, fill_color, matrix)
  def _draw_line_path_impl(self, points, stroke_color, stroke_weight, cap_detail, cap=_GraphicsConstants.PROJECT, matrix:Matrix = None, closed=False):
    """
    Draws a series of connected lines (a path) with specified stroke color and weight.

    If closed is True, it connects the last point back to the first point.

    :param points: A list of tuples representing the points in the path.
    :param stroke_color: The color to use for the stroke.
    :param stroke_weight: The weight/thickness of the stroke lines.
    :param cap_detail: The level of detail used when rendering line caps.
    :param closed: Whether to close the path by connecting the last point to the first.
    :param cap: The cap style to use for rendering line ends. Defaults to PROJECT.
    """
    # Initialize an empty list to hold lists of points for each shape
    
    if not self._do_stroke:
      return
    
    if not points:
      return
    
    if len(points) == 1:
      points.append((points[0][0] + 0.001, points[0][1]))
    
    cap_polys = []
    
    # Iterate over each line segment
    for i in range(len(points) - 1):
      # Generate cap points for the current segment
      cap_points = cap.points(points[i], points[i + 1], stroke_weight, cap_detail)
      # Append the list of cap points as a single item to cap_polys
      cap_polys.append(cap_points)
    
    # If the path is closed, connect the last point to the first
    if closed:
      cap_points = cap.points(points[-1], points[0], stroke_weight, cap_detail)
      cap_polys.append(cap_points)
    
    # Now, cap_polys is a list of lists, where each sublist represents a shape
    for shape in cap_polys:
      self._draw_shape_impl(shape, stroke_color, matrix)
  #TODO: OpenGL for image
  def _image_pygame_impl(self, graphics, x1, y1, x2, y2, image_mode):
    self.finalize()
    tx, ty, r, sx, sy = self._last_matrix.decompose()
    
    x1, y1, x2, y2 = image_mode.corners(x1, y1, x2, y2)
    width, height = (x2 - x1) * sx, (y2 - y1) * sy
    
    desired_coords = [self._last_matrix @ p for p in [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]]
    min_x = min(x for x, _ in desired_coords)
    min_y = min(y for _, y in desired_coords)
    
    if isinstance(graphics, Graphics):
      graphics.finalize()
    
    scaled = pygame.transform.scale(graphics, (width, height))
    rotated = pygame.transform.rotate(scaled, r)
    
    l = int(min_x)
    t = int(min_y)
    r = int(graphics.get_width())
    b =  int(graphics.get_height())
    rect = pygame.Rect(l, t, r, b)
    
    self.blit(rotated, rect)
    
  def _image_opengl_impl(self, graphics, x1, y1, x2, y2, image_mode):
    tx, ty, r, sx, sy = self._last_matrix.decompose()

    # Enable blending for alpha transparency
    Graphics._gl.glEnable(Graphics._gl.GL_BLEND)
    Graphics._gl.glBlendFunc(Graphics._gl.GL_SRC_ALPHA, Graphics._gl.GL_ONE_MINUS_SRC_ALPHA)

    # Bind the texture
    Graphics._gl.glBindTexture(Graphics._gl.GL_TEXTURE_2D, graphics._texture)

    # Apply the transformation matrix from the Matrix object
    transform_matrix = self._last_matrix.to_opengl()
    Graphics._gl.glPushMatrix()
    Graphics._gl.glMultMatrixf(transform_matrix)

    # Draw a textured quad
    Graphics._gl.glBegin(Graphics._gl.GL_QUADS)
    Graphics._gl.glTexCoord2f(0, 0); Graphics._gl.glVertex2f(x1, y1)
    Graphics._gl.glTexCoord2f(1, 0); Graphics._gl.glVertex2f(x2, y1)
    Graphics._gl.glTexCoord2f(1, 1); Graphics._gl.glVertex2f(x2, y2)
    Graphics._gl.glTexCoord2f(0, 1); Graphics._gl.glVertex2f(x1, y2)
    Graphics._gl.glEnd()

    # Restore the previous matrix state
    Graphics._gl.glPopMatrix()

    # Unbind the texture
    Graphics._gl.glBindTexture(Graphics._gl.GL_TEXTURE_2D, 0)

    # Disable blending if it's not needed for subsequent drawing operations
    Graphics._gl.glDisable(Graphics._gl.GL_BLEND)

  def _image_impl(self, graphics, x1, y1, x2, y2, image_mode):
    if self._uses_gl:
      self._image_opengl_impl(graphics, x1, y1, x2, y2, image_mode)
    else:
      self._image_pygame_impl(graphics, x1, y1, x2, y2, image_mode)
  
  def _pygame_fill(self, color, rect=None, special_flags=0):
    """
    Wrapper for the pygame.Surface.fill() method.

    :param color: The color to fill with.
    :param rect: Optional rectangle area to fill.
    :param special_flags: Optional blend mode flags.
    """
    super().fill(color, rect, special_flags)
  
  @staticmethod
  def color(a, b=None, c=None, d=None, is_hsba:bool=False):
    _is_hsba = is_hsba
    return Color.color(a, b, c, d, is_hsba=_is_hsba)
  
      
class Sketch(ABC):
  _instance = None

  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super(Sketch, cls).__new__(cls)
    return cls._instance

  def __init__(self, width, height, framerate=60, use_gl:bool = False):
    if not hasattr(self, 'initialized'):  # This ensures __init__ is only called once
      pygame.init()
      self._g = Graphics(width, height, use_gl)
      self._uses_gl = use_gl
      self._screen = pygame.display.set_mode((width, height))
      self._framerate = framerate
      
      self._frame = 0
      self._mouse_x = 0
      self._mouse_y = 0
      self._last_mouse_x = 0
      self._last_mouse_y = 0
      self._mouse_button = -1
      self._mouse_buttons = []
      self._key = ''
      self._key_code = 0
      self._keys = []
      self._key_codes = []
      self._width = self._screen.get_width()
      self._height = self._screen.get_height()
      self._last_width = self._width
      self._last_height = self._height
      self._pixel_density = 1
      
      self.initialized = True
      self.running = False
      self.run()
      
  @abstractmethod
  def setup(self):
    pass
  @abstractmethod
  def draw(self, graphics):
    pass
  def mouse_pressed(self):
    pass
  def mouse_released(self):
    pass
  def mouse_moved(self):
    pass
  def mouse_dragged(self):
    pass
  def mouse_wheel(self, count):
    pass
  def key_pressed(self):
    pass
  def key_released(self):
    pass
  def quit(self):
    pass
  
  def run(self):
    if self.running:
      return
    
    self.running = True
    
    self.setup()
    
    # Main game loop
    while self.running:
      # Collect all events
      events = pygame.event.get()
      pygame.event.pump()
      
      # Process individual events
      for event in events:
        if event.type == pygame.QUIT:
          self.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
          self._mouse_button = event.button
          self._mouse_buttons.append(self._mouse_button)
          self.mouse_pressed()
        elif event.type == pygame.MOUSEBUTTONUP:
          self._mouse_button = event.button
          self._mouse_buttons.append(self._mouse_button)
          self.mouse_released()
        elif event.type == pygame.MOUSEMOTION:
          self._last_mouse_x, self._last_mouse_y = self._mouse_x, self._mouse_y
          self._mouse_x, self._mouse_y = event.pos
          if any(event.buttons):  # Left mouse button is pressed
            self.mouse_dragged()
          else:
            self.mouse_moved()
        elif event.type == pygame.MOUSEWHEEL:
          self.mouse_wheel(event.y)
        elif event.type == pygame.KEYDOWN:
          self._key = event.unicode
          self._key_code = event.key
          self._keys.append(self._key)
          self._key_codes.append(self._key_code)
          self.key_pressed()
        elif event.type == pygame.KEYUP:
          self._key = event.unicode
          self._key_code = event.key
          self._keys.remove(self._key)
          self._key_codes.remove(self._key_code)
          self.key_released()
        elif event.type == pygame.VIDEORESIZE:
          self._screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
          _g = self._g
          self._g = Graphics(event.w * self._pixel_density, event.h * self._pixel_density, self._g._uses_gl)
          self._g.blit(_g, (0, 0))
      
      self.draw(self._g)
      scaled = self._g.resize(self.width, self.height, to_native=True)
      self._screen.fill((0, 0, 0), (0, 0, self._screen.get_width(), self._screen.get_height()))
      self._screen.blit(scaled, (0, 0))
      pygame.display.update()
      pygame.time.Clock().tick(self._framerate)
      
      self._frame += 1
    
    pygame.quit()
  
  @property
  def frame(self):
    return self._frame
  
  @property
  def width(self):
    return self._screen.get_width()
  
  @property
  def height(self):
    return self._screen.get_height()
  
  @property
  def last_width(self):
    return self._last_height # TODO
  
  @property
  def last_height(self):
    return self._last_height # TODO
  
  @property
  def mouse_x(self):
    return self._mouse_x
  
  @property
  def mouse_y(self):
    return self._mouse_y
  
  @property
  def last_mouse_x(self):
    return self._last_mouse_x
  
  @property
  def last_mouse_y(self):
    return self._last_mouse_y
  
  @property
  def mouse_button(self):
    return self._mouse_button
  
  @property
  def mouse_buttons(self):
    return [a for a in self._mouse_buttons]
  
  @property
  def is_mouse_pressed(self):
    return len(self._mouse_buttons) > 0
  
  @property
  def key(self):
    return self._key
  
  @property
  def key_code(self):
    return self._key_code
  
  @property
  def keys(self):
    return [a for a in self._keys]
  
  @property
  def key_codes(self):
    return [a for a in self._key_codes]
  
  @staticmethod
  def get_pypage_key(key_value):
    field_name = 'K_' + key_value
    try:
      return getattr(pygame, field_name)
    except AttributeError:
      return None
  
  @staticmethod
  def pygame_key_code(key):
    return pygame.key.key_code(key)
  
  def exit(self):
    self.quit()
    self.running = False
    Sketch._instance = None
  
  def framerate(self, framerate):
    if not isinstance(framerate, (int, np.integer)):
      raise TypeError("Framerate must be an integer.")
    if framerate < 0:
      raise ValueError("Framerate must be positive.")
    self._framerate = framerate
  
  def set_size(self, x, y):
    if not all(isinstance(arg, (int, np.integer)) for arg in [x, y]):
      raise TypeError("Size must be an integer.")
    if x < 0 or y < 0:
      raise ValueError("Size must be positive.")
    self._screen.set_mode((int(x), int(y)), pygame.RESIZABLE)
    #Assumes this causes an event
  
  def pixel_density(self, pixel_density):
    self._pixel_density = pixel_density
    self._g = self._g.resize(self._width * self._pixel_density, self.height * self._pixel_density)
  
  def use_gl(self, use_gl):
    self._uses_gl = use_gl
    
  def emulate_key_press(self, key, key_code=None):
    if not key_code:
      key_code = pygame.key.key_code(key)
    self._key = key
    self._key_code = key_code
    self._keys.append(self._key)
    self._key_codes.append(self._key_code)
    self.key_pressed()
    
  def emulate_key_release(self, key, key_code=None):
    if not key_code:
      key_code = pygame.key.key_code(key)
    self._key = key
    self._key_code = key_code
    self._keys.remove(self._key)
    self._key_codes.remove(self._key_code)
    self.key_released()