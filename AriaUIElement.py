#Requires Python 3.7
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
import AriaSketch
import time as clock

class UIElement:
  """
  A base class for creating UI elements within a graphical user interface.

  UIElement serves as a blank slate for UI components and is designed to be extended and customized.
  It provides the foundational structure for positioning, transforming, and managing hierarchical
  relationships between UI elements.

  Each UIElement must have a parent to maintain the tree structure of the UI, except for UIRoot,
  which is the root element of the UI tree and does not require a parent.

  To add input functionality, UIElements can implement UIMouseListener and UIKeyboardListener interfaces.
  These interfaces allow UIElements to respond to mouse and keyboard events.

  A UIElement can have its own AriaSketch.Graphics context, which can be enabled or disabled using the method
  `self.use_own_graphics(bool)`. This allows for more control over the rendering of individual elements.

  Methods in this class are designed to be chainable, providing a fluent interface for setting properties
  and invoking behaviors. This design choice enhances readability and ease of use when working with UIElements.

  This library relies on the AriaSketch library, which is a port of Java Processing's drawing functionality,
  providing a simple and familiar API for AriaSketch.Graphics operations.

  Attributes:
    ... (attributes related to transform, tree structure, and AriaSketch.Graphics context)

  Methods:
    ... (methods related to lifecycle events, drawing, input handling, and dynamic properties)

  Example:
    class CustomButton(UIElement, UIMouseListener):
      def __init__(self, parent):
        super().__init__(parent)
        self.use_own_graphics(True)
        # Additional initialization for the custom button

      def on_draw(self, AriaSketch.Graphics):
        # Custom drawing code for the button
        pass

      ... (other overridden methods and custom methods)

    Note:
      When creating a new UI hierarchy, start with an instance of UIRoot as the root element.
      Then extend UIElement to create custom components that make up your interface.
      Remember to chain methods where appropriate for a cleaner and more intuitive API usage.
  """
  _roots = []

  def __init__(self, parent=None):
    self._x, self._y, self._r, self._sx, self._sy, self._ax, self._ay = 0, 0, 0, 1, 1, 0, 0
    self._local_matrix, self._global_matrix, self._reversed_local_matrix, self._reversed_global_matrix = None, None, None, None
    self._update_local_matrix, self._update_global_matrix = False, False
    
    self._width, self._height = 1, 1
    self._rebuild_graphics, self._own_graphics = False, False
    self._graphics = None
    
    self._is_destroyed = False
    self._parent = None
    self._children = []
    
    self._render_time, self._tree_render_time = 0, 0

    if isinstance(self, UIRoot):
      UIElement._roots.append(self)
    
    self.set_parent(parent)
    self.set_transform(0, 0, 0, 1, 1, 0, 0)
    self.set_graphics_size(1, 1)
    self.request_graphics_rebuild()
    self.request_global_matrix_update()
    self.request_local_matrix_update()

  def __str__(self):
    return str(type(self).__name__)


  #Roots
  @staticmethod
  def getRoots():
    return [arg for arg in UIElement._roots]
  


  #Transform
  #TODO: Add setTransform variations and getters
  def set_transform(self, x, y, r, sx, sy, ax, ay) -> UIElement:
    if self._is_destroyed:
      return self
    
    ax = max(0, min(1, ax))
    ay = max(0, min(1, ay))
    
    if x == self._x and y == self._y and r == self._r and sx == self._sx and sy == self._sy and ax == self._ax and ay == self._ay:
      return self
    
    olx, oly, olr, olsx, olsy, olax, olay = self._x, self._y, self._r, self._sx, self._sy, self._ax, self._ay
    self._x, self._y, self._r, self._sx, self._sy, self._ax, self._ay =  x, y, r, sx, sy, ax, ay
    self.request_local_matrix_update()
    self.on_transform_changed(olx, oly, olr, olsx, olsy, olax, olay)
  
    return self
    
  def set_transform_from_element(self, element) -> UIElement:
    return self.set_transform(element._x, element._y, element._r, element._sx, element._sy, element._ax, element._ay)

  def set_position(self, x, y) -> UIElement:
    return self.set_transform(x, y, self._r, self._sx, self._sy, self._ax, self._ay)

  def set_rotation(self, r) -> UIElement:
    return self.set_transform(self._x, self._y, r, self._sx, self._sy, self._ax, self._ay)

  def set_scale(self, sx, sy) -> UIElement:
    return self.set_transform(self._x, self._y, self._r, sx, sy, self._ax, self._ay)

  def set_align(self, ax, ay) -> UIElement:
    return self.set_transform(self._x, self._y, self._r, self._sx, self._sy, ax, ay)
  
  def get_position_x(self) -> float:
    return self._x

  def get_position_y(self) -> float:
    return self._y

  def get_rotation(self) -> float:
    return self._r
  
  def get_scale_x(self) -> float:
    return self._sx
  
  def get_scale_y(self) -> float:
    return self._sy

  def get_align_x(self) -> float:
    return self._ax

  def get_align_y(self) -> float:
    return self._ay

  def request_local_matrix_update(self) -> UIElement:
    self._update_local_matrix = True
    self._update_global_matrix = True
    return self

  def request_global_matrix_update(self) -> UIElement:
    self._update_global_matrix = True
    return self

  def is_global_matrix_update_requested(self) -> bool:
    if self._is_destroyed:
      return False
    if self._update_global_matrix:
      return True
    if self.is_root():
      return False
    return self._parent.is_global_matrix_update_requested()

  def update_matrices(self) -> UIElement:
    if self.is_destroyed():
      return self
      
    event = False
    local_matrix = self._local_matrix.copy() if self._local_matrix else None
    global_matrix = self._global_matrix.copy() if self._local_matrix else None
    
    if self._update_local_matrix:
      alx, aly = - self._ax * self._width * self._sx, - self._ay * self._height * self._sy
      sin = np.sin(self._r);
      cos = np.cos(self._r);
      self._local_matrix = AriaSketch.Matrix.identity()
      self._local_matrix = self._local_matrix.translate(self._x, self._y)
      self._local_matrix = self._local_matrix.rotate(self._r)
      self._local_matrix = self._local_matrix.translate(alx, aly)
      self._local_matrix = self._local_matrix.scale(self._sx, self._sy);
      
      self._reversed_local_matrix = self._local_matrix.invert()
      
      self._update_local_matrix = False;
      self._update_global_matrix = True;
      event = True
    if self.is_global_matrix_update_requested():
      if self.is_root():
        self._global_matrix = self._local_matrix.copy()
      else:
        self._parent.update_matrices();
        self._global_matrix = self._parent._global_matrix.copy()
        self._global_matrix = self._global_matrix @ self._local_matrix
      self._reversed_global_matrix = self._global_matrix.invert()
      
      self._update_global_matrix = False;
      for child in self._children:
        child.request_global_matrix_update();
      event = True
    
    if event:
      self.on_matrix_update(local_matrix, global_matrix);
    
    return self

  def get_local_matrix(self) -> AriaSketch.Matrix:
    return self.update_matrices()._local_matrix.copy()

  def get_global_matrix(self) -> AriaSketch.Matrix:
    return self.update_matrices()._global_matrix.copy()

  def get_reversed_local_matrix(self) -> AriaSketch.Matrix:
    return self.update_matrices()._reversed_local_matrix.copy()

  def get_reversed_global_matrix(self) -> AriaSketch.Matrix:
    return self.update_matrices()._reversed_global_matrix.copy()

  def global_to_local(self, points, y=None):
    if all(isinstance(arg, (int, float, np.integer, np.floating)) for arg in [points, y]):
      pt = self.get_reversed_global_matrix() @ (points, y)
      return pt
    return self.get_global_matrix() @ points

  def local_to_global(self, points, y=None):
    if all(isinstance(arg, (int, float, np.integer, np.floating)) for arg in [points, y]):
      pt = self.get_global_matrix() @ (points, y)
      return pt
    return self.get_reversed_global_matrix() @ points

  def get_bounding_box(self, root=None):
    if root is None:
      root = self.get_root()
    if root is None:
      return None
    
    self.update_matrices()
    tl = root.global_to_local(self.local_to_global(0, 0))
    tr = root.global_to_local(self.local_to_global(self._width, 0))
    rb = root.global_to_local(self.local_to_global(self._width, self._height))
    lb = root.global_to_local(self.local_to_global(0, self._height))
    
    lx = min(tl[0], tr[0], lb[0], rb[0])
    ly = min(tl[1], tr[1], lb[1], rb[1])
    hx = max(tl[0], tr[0], lb[0], rb[0])
    hy = max(tl[1], tr[1], lb[1], rb[1])
    
    return lx, ly, hx, hy
  
  def get_subtree_bounding_box(self, include_self=True, root=None):
    lx, ly, hx, hy = float('inf'), float('inf'), float('-inf'), float('-inf')
    
    for child in self._children:
      minx, miny, maxx, maxy = child.get_subtree_bounding_box(True, root)
      lx, ly, hx, hy = min(lx, minx), min(ly, miny), max(hx, maxx), max(hy, maxy)
    
    if include_self:
      minx, miny, maxx, maxy = self.get_bounding_box(root)
      lx, ly, hx, hy = min(lx, minx), min(ly, miny), max(hx, maxx), max(hy, maxy)
    
    return lx, ly, hx, hy
    

  def is_local_point_within_rectangle(self, x, y, global_arguments=False) -> bool:
    if global_arguments:
      x, y = self.global_to_local(x, y)
    return x >= 0 and x < self.get_graphics_width() and y >= 0 and y < self.get_graphics_height()
  
  
  
  #AriaSketch.Graphics
  def get_graphics_width(self) -> int:
    return self._width

  def get_graphics_height(self) -> int:
    return self._height

  def set_graphics_size(self, width, height) -> UIElement:
    if self.is_destroyed():
      return self
      
    if width < 1:
      width = 1
    if height < 1:
      height = 1
    if width == self._width and height == self._height:
      return self
    
    old_width = self._width
    old_height = self._height
    
    self._width = width
    self._height = height
    
    self.request_local_matrix_update()
    self.request_graphics_rebuild()
    self.on_graphics_size_changed(old_width, old_height)
    
    return self

  def request_graphics_rebuild(self) -> UIElement:
    self._rebuild_graphics = True
    return self

  def uses_own_graphics(self) -> bool:
    return self._own_graphics

  def use_own_graphics(self, own_raphics) -> UIElement:
    self._own_graphics = own_raphics
    return self

  def rebuild_graphics(self) -> UIElement:
    if not self._rebuild_graphics or not self._own_graphics:
      self._rebuild_graphics = False
      return self
    
    self._graphics = AriaSketch.Graphics(self.get_graphics_width(), self.get_graphics_height(), use_gl=True)
    self._rebuild_graphics = False
    
    return self
  
  #TODO: Test
  def draw(self, graphics:AriaSketch.Graphics):
    if self.is_destroyed():
      return self
    if graphics is None:
      return
    
    start_time = int(clock.time() * 1000)
    
    self.on_pre_draw(graphics)
    self.update_matrices()
    
    graphics.push_matrix()
    graphics.apply(self._local_matrix)
    
    self.rebuild_graphics()
    
    if self._own_graphics:
      self_time = int(clock.time() * 1000)
      self.on_draw(self._graphics)
      self._render_time = int(clock.time() * 1000) - self_time
      
      for child in self._children:
        child.draw(self._graphics)
      
      self.on_post_draw(self._graphics)
      
      graphics.image_mode(AriaSketch.Graphics.CORNER)
      graphics.image(self._graphics, 0, 0)
      
    else:
      self_time = int(clock.time() * 1000)
      self.on_draw(graphics)
      self._render_time = int(clock.time() * 1000) - self_time

      for child in self._children:
        child.draw(graphics)
      
      self.on_post_draw(graphics)
    
    graphics.pop_matrix()
    self._tree_render_time = int(clock.time() * 1000) - start_time
    
    return self

  def _get_own_graphics(self):
    return self._graphics
  
  
  
  
  
  #Tree
  def destroy(self) -> UIElement:
    if self.is_destroyed():
      return self
    
    self._is_destroyed = True
    
    for child in self._children:
      child.destroy()
      
    if self.is_root():
      UIElement.roots.remove(self)
    
    if self._parent is not None:
      self._parent.children.remove(self)
    
    previous_parent = self._parent
    self._parent = None
    self.on_destroyed(previous_parent)
    
    return self

  def is_destroyed(self) -> bool:
    return self._is_destroyed

  def is_root(self) -> bool:
    return isinstance(self, UIRoot)

  def get_root(self) -> UIRoot:
    if self.is_destroyed():
      return None
    if self.is_root():
      return self
    return self._parent.get_root()

  def get_parent(self) -> UIElement:
    return self._parent

  def get_level(self) -> int:
    if self.is_root():
      return 0
    return self._parent.get_level() + 1

  def get_depth(self) -> int:
    if len(self._children) == 0:
      return 0
    return max(child.get_depth() for child in self._children) + 1
    
  def is_descendant_of(self, element:UIElement) -> bool:
    if element.is_destroyed():
      return False
    return element.is_ancestor_of(self)
    
  def is_ancestor_of(self, element:UIElement) -> bool:
    if element is None:
      return False
      
    while element is not None:
      if element is self:
        return True
      element = element._parent
    
    return False
  
  @staticmethod
  def common_ancestor(*elements) -> UIElement:
    if len(elements) == 0:
      return None
    if len(elements) == 1:
      return elements[0]
    if len(elements) == 2:
      return elements[0].get_common_ancestor(elements[1])
    
    current = elements[0]
    for i in range(1, len(elements)):
      current = elements[i].get_common_ancestor(current)
      if current is None:
        return None
    
    return current

  def get_common_ancestor(self, element:UIElement) -> UIElement:
    if element is None:
      return None
    
    ancestors1 = set()
    
    s = self
    
    # Traverse up the tree for obj1
    while element:
      ancestors1.add(element)
      element = element.get_parent()
    
    # Traverse up the tree for obj2 and check for common ancestor
    while s:
      if s in ancestors1:
        return s  # Found the common ancestor
      s = s.get_parent()
    
    return None  # No common ancestor found
    

  def set_parent(self, parent:UIElement, index:int=None) -> UIElement:
    if self.is_destroyed():
      return self
    
    if isinstance(self, UIRoot):
      if parent is None:
        return self
      raise RuntimeError("Root cannot have a parent.")
    
    if self.is_ancestor_of(parent):
      return self
    
    previous_parent = self._parent
    previous_index = -1 if self._parent is None else self._parent._children.indexof(self)
    if index is None and parent is not None:
      index = parent.get_child_count()
    
    if parent is None:
      #Try to set root even if already root
      if self._parent is None:
        raise RuntimeError("UIElement cannot be root, unless it is Root.")
      
      #Destroy
      self.destroy()
    else:
      #Set parent (for the first time)
      index = 0 if index < 0 else parent.get_child_count() if index > parent.get_child_count() else index
      
      if self._parent is None:
        can_be_child_of = self.can_be_child_of(parent)
        can_be_parent_of = parent.can_be_parent_of(self, index)
        if not can_be_child_of or not can_be_parent_of:
          if not can_be_parent_of:
            parent.on_child_rejected(self, index)
          if can_be_parent_of:
            self.on_parent_rejected(parent, index)
          if self._parent is None:
            raise RuntimeError("Parent couldn't be set.")
          return self
        
        
        self._parent = parent
        
        self.request_global_matrix_update()
        
        self._parent._children.insert(index, self)
        self.on_parent_set(previous_parent)
        self._parent.on_child_added(self, index)
        
      #Change parent's child index
      elif self._parent is parent:
        index = 0 if index < 0 else parent.get_child_count() if index > parent.get_child_count() else index
        
        if index == previous_index:
          return self #Index didn't change
        if not self._parent.can_child_index_change(self, previous_index, index):
          return self
        #TODO: onChangeIndexRejected
        self._parent._children.pop(previous_index)
        self._parent._children.insert(index, self)
        self._parent.on_child_changed_index(self, previous_index, index)
      #Change parent
      else:
        index = 0 if index < 0 else parent.get_child_count() if index > parent.get_child_count() else index
        if index == previous_index:
          return self #Index didn't change
        if not self._parent.can_child_be_removed(self, previous_index) or not parent.can_be_parent_of(self, index) or not self.can_parent_be_removed() or not self.can_be_child_of(parent):
          #TODO: onChildRejected
          return self
        
        self._parent._children.pop(previous_index)
        self._parent = parent
        self._parent._children.insert(index, self)
        
        self.request_global_matrix_update()
        
        previous_parent.on_child_removed(self, previous_index)
        self._parent.on_child_added(self, index)
        self.on_parent_set(previous_parent)
    
    return self

  def add_child(self, child, index=None) -> UIElement:
    if child:
      child.set_parent(self, index)
    
  def add_children(self, index=None, *children) -> UIElement:
    for child in children:
      self.add_child(child, index)
  
  def remove_child(self, child):
    if isinstance(child, UIElement):
      if child._parent is not self:
        return self
      child.set_parent(None)
      return self
    else:
      if child < 0 or child == len(self._children):
        return self
      self._children[child].set_parent(None)
      return self
  def remove_children(self, *children) -> UIElement:
    for child in children:
      self.remove_child(child)

  def index_of(self, child) -> int:
    return self._children.index(child)

  def get_child(self, index) -> UIElement:
    return self._children[index]

  def get_children(self) -> list:
    return [arg for arg in self._children]

  def get_child_count(self) -> int:
    return len(self._children)
  
  def search(self, *search_criteria):
    """
    Searches for UIElements that match all provided search criteria.

    This method collects all descendant UIElements of the current element and filters them based on the provided UISearchCriteria objects.

    Parameters:
        *search_criteria (UISearchCriteria): A variable number of UISearchCriteria objects whose filter methods are used to determine if an element matches the desired criteria.

    Returns:
        list: A list of UIElements that match all the provided search criteria.
    """
    
    candidates = [self]
    
    # Retrieve all UIElements
    index = 0
    while index < len(candidates):
      candidates.extend(candidates[index].get_children())
      index += 1
    
    # Filter candidates based on each search criterion
    for criteria in search_criteria:
      candidates = [candidate for candidate in candidates if criteria.filter(candidate)]
    
    return candidates
  
  
  #Diagnostics
  def get_render_time(self) -> int:
    return self._render_time

  def get_tree_render_time(self) -> int:
    return self._tree_render_time

  def draw_bounding_box(self, graphics:AriaSketch.Graphics, root:UIElement=None) -> UIElement:
    l, t, r, b = self.get_bounding_box(root)
    self.on_draw_bounding_box(graphics, self.get_level(), l, t, r, b)
    
    for child in self._children:
      child.draw_bounding_box(graphics)
      
    return self

  def on_draw_bounding_box(self, graphics:AriaSketch.Graphics, level, left, top, right, bottom) -> None:
    hue = level * 0.125
    
    r = abs((6 * hue) % 6 - 3) - 1
    g = 2 - abs((6 * hue) % 6 - 2)
    b = 2 - abs((6 * hue) % 6 - 4)
    
    r = AriaSketch.Color.byte(r * 255)
    g = AriaSketch.Color.byte(g * 255)
    b = AriaSketch.Color.byte(b * 255)
    
    color = AriaSketch.Color.color(r, g, b)
    
    graphics.no_fill()
    graphics.stroke_weight(1)
    graphics.stroke(color)
    graphics.rect_mode(AriaSketch.Graphics.CORNERS)
    graphics.rect(left, top, right, bottom)
    
    graphics.fill(color)
    graphics.text_size(10)
    graphics.text_align(0, 0)
    graphics.text(str(self), left, top + 10 * self.get_level())
  
  
  
  #Events and Dynamic Properties
  #----Transforms
  def on_transform_changed(self, old_x, old_y, old_r, old_sx, old_sy, old_ax, old_ay) -> None:
    pass

  def on_matrix_update(self, oldLocalmatrix, oldGlobalMatrix) -> None:
    pass

  def on_graphics_size_changed(self, oldWidth, oldHeight) -> None:
    pass

  def on_pre_draw(self, graphics:AriaSketch.Graphics) -> None:
    pass

  def on_draw(self, graphics:AriaSketch.Graphics) -> None:
    pass

  def on_post_draw(self, graphics:AriaSketch.Graphics) -> None:
    pass

  def is_point_interactable(self, x, y) -> bool:
    return True

  #----Tree (Child)
  def can_be_child_of(self, parent) -> bool:
    return True

  def can_parent_be_removed(self) -> bool:
    return True

  #----Tree (Parent)
  def can_be_parent_of(self, child, index) -> bool:
    return True

  def can_child_index_change(self, parent, before, after) -> bool:
    return True

  def can_child_be_removed(self, child, index) -> bool:
    return True

  #----Tree (Events)
  def on_child_destroyed(self, child, index) -> None:
    pass

  def on_destroyed(self, previous_parent) -> None:
    pass

  def on_parent_set(self, previous_parent) -> None:
    pass

  def on_parent_rejected(self, parent, index) -> None:
    pass

  def on_child_added(self, child, index) -> None:
    pass

  def on_child_rejected(self, child, index) -> None:
    pass

  def on_child_removed(self, child, index) -> None:
    pass

  def on_child_changed_index(self, child, previous_index, index) -> None:
    pass
  
class UIRoot(UIElement):
  def __init__(self, sketch:AriaSketch.Sketch):
    super().__init__(None)
    self._sketch = sketch
    #TODO
    pass

  @property
  def sketch(self) -> AriaSketch.Sketch:
    return self._sketch
    
  def get_pointed_at_element(self, x, y, excluded_objects=None, excluded_classes=None, included_classes=None, search_criteria=None):
    # Create a PointInteractableSearchCriteria object with the given parameters
    point_criteria = _PointInteractableSearchCriteria(x, y, excluded_objects, excluded_classes, included_classes)
        
    # Combine the point criteria with any additional user-specified criteria
    # Initialize all_criteria with point_criteria
    all_criteria = [point_criteria]
    
    # Extend all_criteria with any additional user-specified criteria if provided
    if search_criteria:
      all_criteria.extend(search_criteria)
    
    # Use the search method with all criteria to find matching elements
    pointed_elements = self.search(*all_criteria)
        
    if pointed_elements:
      return pointed_elements[-1]
    
    return None


class UIMouse:
  def __init__(self, root:UIRoot):
    if root is None:
      raise TypeError("Can't init Mouse, root is null.")
    self._root = root

    self._pressed = None
    self._lastPressed = None

    self._start = None
    self._last_pressed_left = None
    self._last_pressed_right = None
    self._last_pressed_center = None
    self._last_pressed_alt = None

    self._last = None
    self._current = None
  
  #Event Handlers
  def mouse_pressed(self, x, y, button) -> None:
    button = UIMouseEvent.mouse_button(button)
    event = UIMouseEvent(x, y, 0, button | UIMouseEvent.PRESSED, (self._current._down | button) if self._current else button)
    pointed = self._root.get_pointed_at_element(x, y, included_classes=UIMouseListener)
    
    if pointed is None:
      return
      
    if self._start is None:
      self._start = event
      
    if button == UIMouseEvent.LEFT:
      self._last_pressed_left = event
    if button == UIMouseEvent.RIGHT:
      self._last_pressed_right = event
    if button == UIMouseEvent.CENTER:
      self._last_pressed_center = event
    if button == UIMouseEvent.ALT:
      self._last_pressed_alt = event
      
    if self._current is None:
      self._current = event
    self._last = self._current
    self._current = event
      
    self._pressed = pointed
    if self._pressed is not None:
      self._pressed.on_mouse_pressed(self, self._current.local(self._pressed))
  
  def mouse_released(self, x, y, button) -> None:
    button = UIMouseEvent.mouse_button(button)
    event = UIMouseEvent(x, y, 0, button | UIMouseEvent.RELEASED, (self._current._down & ~ button) if self._current else 0)
    
    if self._start is None:
      self._start = event
    
    
    if button == UIMouseEvent.LEFT:
      self._last_pressed_left = None
    if button == UIMouseEvent.RIGHT:
      self._last_pressed_right = None
    if button == UIMouseEvent.CENTER:
      self._last_pressed_center = None
    if button == UIMouseEvent.ALT:
      self._last_pressed_alt = None
    
    if self._current is None:
      self._current = event
    self._last = self._current
    self._current = event
    
    released = self._pressed
    if not self._current.any_button_down():
      self._pressed = None
    if released is not None:
      released.on_mouse_released(self, self._current.local(released))
      
    if not self._current.any_button_down():
      self._start = None

  def mouse_moved(self, x, y) -> None:
    event = UIMouseEvent(x, y, 0, 0, self._current._down) if self._current else UIMouseEvent(self._current, x, y, 0, 0)
    if self._current is None:
      self._current = event
    self._last = self._current
    self._current = event
      
    if self._pressed is None:
      pointed = self._root.get_pointed_at_element(x, y, included_classes=[UIMouseListener])
      if pointed is None:
        return
      pointed_listener = pointed
      pointed_listener.on_mouse_moved(self, self._current)
    else:
      self._pressed.on_mouse_moved(self, self._current.local(self._pressed))

  def mouse_wheel(self, x, y, scroll) -> None:
    #TODO: Might be wrong
    event = UIMouseEvent(x, y, scroll, UIMouseEvent.WHEEL, 0) if self._current else UIMouseEvent(self._current, x, y, scroll, UIMouseEvent.WHEEL)
    if self._current is None:
      self._current = event
    self._last = self._current
    self._current = event
      
    if self._pressed is None:
      pointed = self._root.get_pointed_at_element(x, y, UIMouseListener)
      if pointed is None:
        return
      pointed_listener = pointed
      pointed_listener.on_mouse_wheel(self, self._current.local(pointed))
    else:
      self._pressed.on_mouse_wheel(self, self._current)
  
  def get_pressed(self) -> UIElement:
    return self._pressed
  
  #Event getters
  def get_start(self) -> UIMouseEvent:
    return self._start

  def get_last_pressed_left(self) -> UIMouseEvent:
    return self._last_pressed_left

  def get_last_pressed_right(self) -> UIMouseEvent:
    return self._last_pressed_right

  def get_last_pressed_center(self) -> UIMouseEvent:
    return self._last_pressed_center

  def get_last_pressed_alt(self) -> UIMouseEvent:
    return self._last_pressed_alt

  def get_last(self) -> UIMouseEvent:
    return self._last

  def get_current(self) -> UIMouseEvent:
    return self._current
  
  #Value getters
  def get_x(self) -> float:
    return self._current._x if self._current else None

  def get_y(self) -> float:
    return self._current._y if self._current else None

  def is_pressed(self) -> bool:
    return self._current.any_button_down() if self._current else None


  def is_left_down(self) -> bool:
    return self._current.is_left_down() if self._current else None

  def is_right_down(self) -> bool:
    return self._current.is_right_down() if self._current else None

  def is_center_down(self) -> bool:
    return self._current.is_center_down() if self._current else None

  def is_alt_down(self) -> bool:
    return self._current.is_alt_down() if self._current else None

class UIMouseEvent:
  LEFT =     1
  RIGHT =    2
  CENTER =   4
  ALT =      8
  WHEEL =    16
  PRESSED =  32
  RELEASED = 64
  
  BUTTON_MASK = 0xF
  TYPE_MASK = 0x70
  
  PYGAME_LEFT = 1
  PYGAME_RIGHT = 3
  PYGAME_CENTER = 2

  def __init__(self, x, y, scroll, event, down, time=None):
    self._x = x
    self._y = y
    self._scroll = scroll
    self._time = clock.time() if time is None else time
    self._event = event
    self._down = down
    
  @property
  def x(self):
    return self._x
  
  @property
  def y(self):
    return self._y
  
  @property
  def time(self):
    return self._time
  
  @property
  def scroll(self):
    return self._scroll
  
  def __str__(self):
    event = "Event "
    if self._event & UIMouseEvent.PRESSED != 0:
      event += "Pressed "
    if self._event & UIMouseEvent.RELEASED != 0:
      event += "Released "
    if self._event & UIMouseEvent.WHEEL != 0:
      event += "Wheel "
    
    if self._event & UIMouseEvent.WHEEL:
      event += str(self._scroll)
    else:
      if self._event & UIMouseEvent.LEFT != 0:
        event += "Left "
      if self._event & UIMouseEvent.RIGHT != 0:
        event += "Right "
      if self._event & UIMouseEvent.CENTER != 0:
        event += "Center "
      if self._event & UIMouseEvent.ALT != 0:
        event += "Alt "
      event += str(self._x) + " " + str(self._y)
    
    down = "Down "
    if self._down & UIMouseEvent.PRESSED != 0:
      down += "Pressed "
    if self._down & UIMouseEvent.RELEASED != 0:
      down += "Released "
    if self._down & UIMouseEvent.WHEEL != 0:
      down += "Wheel "
    
    if self._down & UIMouseEvent.WHEEL:
      down += str(self._scroll)
    else:
      if self._down & UIMouseEvent.LEFT != 0:
        down += "Left "
      if self._down & UIMouseEvent.RIGHT != 0:
        down += "Right "
      if self._down & UIMouseEvent.CENTER != 0:
        down += "Center "
      if self._down & UIMouseEvent.ALT != 0:
        down += "Alt "
    if down == "Down ":
      down += "None"
    
    return "Mouse Event: " + event + ", " + down
      
    
  def is_invalid(self) -> bool:
    type = self._event & UIMouseEvent.TYPE_MASK
    button = self._event & UIMouseEvent.BUTTON_MASK
        
    #Multiple types
    if (type & (type - 1)) != 0:
      return True
        
    #Mouse button specified on wheel event
    if type == UIMouseEvent.WHEEL and button != 0:
      return True
        
    #Scroll not specified on wheel event
    if type == UIMouseEvent.WHEEL and self._scroll == 0:
      return True
        
    #Scroll specified on pressed/released event
    if (type == UIMouseEvent.PRESSED or type == UIMouseEvent.RELEASED) and self._scroll != 0:
      return True
        
    #Mouse button not specified on pressed/released event
    if (type == UIMouseEvent.PRESSED or type == UIMouseEvent.RELEASED) and button == 0:
      return True
        
    return False
  
  #Event type
  def is_pressed_event(self) -> bool:
    return (self._event & UIMouseEvent.PRESSED) == UIMouseEvent.PRESSED

  def is_released_event(self) -> bool:
    return (self._event & UIMouseEvent.PRESSED) == UIMouseEvent.RELEASED

  def is_wheel_event(self) -> bool:
    return (self._event & UIMouseEvent.PRESSED) == UIMouseEvent.WHEEL

  def is_move_event(self) -> bool:
    return (self._event & UIMouseEvent.PRESSED) == 0

  
  #Event button
  def is_left_event(self) -> bool:
    return (self._event & UIMouseEvent.LEFT) > 0

  def is_right_event(self) -> bool:
    return (self._event & UIMouseEvent.RIGHT) > 0

  def is_center_event(self) -> bool:
    return (self._event & UIMouseEvent.CENTER) > 0

  def is_alt_event(self) -> bool:
    return (self._event & UIMouseEvent.ALT) > 0


  
  #Buttons down
  def is_left_down(self) -> bool:
    return (self._down & UIMouseEvent.LEFT) > 0

  def is_right_down(self) -> bool:
    return (self._down & UIMouseEvent.RIGHT) > 0

  def is_center_down(self) -> bool:
    return (self._down & UIMouseEvent.CENTER) > 0

  def is_alt_down(self) -> bool:
    return (self._down & UIMouseEvent.ALT) > 0

  def any_button_down(self) -> bool:
    return (self._down & UIMouseEvent.BUTTON_MASK) != 0

  def local(self, element:UIMouseListener) -> UIMouseEvent:
    coords = element.global_to_local(self._x, self._y)
    return UIMouseEvent(coords[0], coords[1], self._scroll, self._event, self._down, time=self._time)
  
  def distance_from_sq(self, event, y=None) -> float:
    if isinstance(event, UIMouseEvent):
      xx = self._x - event._x
      yy = self._y - event._y
      return xx * xx + yy * yy
    else:
      xx = self._x - event
      yy = self._y - y
      return xx * xx + yy * yy
  
  def distance_from(self, event, y=None) -> float:
    return np.sqrt(self.distance_from_sq(event, y))
  
  @staticmethod
  def mouse_button(button):
    if button == UIMouseEvent.PYGAME_LEFT:
      return UIMouseEvent.LEFT
    if button == UIMouseEvent.PYGAME_RIGHT:
      return UIMouseEvent.RIGHT
    if button == UIMouseEvent.PYGAME_CENTER:
      return UIMouseEvent.CENTER
    return UIMouseEvent.ALT
  
class UIMouseListener(ABC):
  def on_mouse_pressed(self, mouse:UIMouse, event:UIMouseEvent) -> None:
    pass

  def on_mouse_released(self, mouse:UIMouse, event:UIMouseEvent) -> None:
    pass

  def on_mouse_moved(self, mouse:UIMouse, event:UIMouseEvent) -> None:
    pass

  def on_mouse_wheel(self, mouse:UIMouse, event:UIMouseEvent) -> None:
    pass

  def global_to_local(self, points, y=None) -> tuple:
    pass

class UIKeyboard:
  DIFFERENT_ELEMENT = "A UIElement interupted keyboard interaction."
  
  def __init__(self, root):
    self._root = root
    self._keys = []
    self._current = None
  
  def key_pressed(self, key, keyCode) -> None:
    event = UIKeyboardEvent(key, keyCode)
    
    if event not in self._keys:
      self._keys.append(event)
    
    if self._current:
      if event not in self._keys:
        self._current.on_key_pressed(self, event)
      else:
        self._current.on_key_held(self, event)
    
  def key_released(self, key, keyCode) -> None:
    event = UIKeyboardEvent(key, keyCode)
    
    for k in self._keys:
      if k == event:
        event = UIKeyboardEvent(k._key, k._key_code, last=k, time=event._time)
        self._keys.remove(k)
        if self._current:
          self._current.on_key_released(self, event)
        break
    
  def start_keyboard(self, listener:UIKeyboardListener) -> None:
    if listener is None:
      return
    if not isinstance(listener, UIKeyboardListener):
      return
    
    if self._current is not None:
      if self._current is listener:
        return
      last = self._current
      self._current = listener
      last.on_keyboard_interupted(self, self.get_keys_pressed(), UIKeyboard.DIFFERENT_ELEMENT)
      self._current.on_keyboard_started(self)
      return
    self._current = listener
    self._current.on_keyboard_started(self)
    
  def end_keyboard(self) -> None:
    if self._current is None:
      return
    last = self._current
    self._current.on_keyboard_ended(self)
    self._current = None

  def interupt_keyboard(self) -> None:
    if self._current is None:
      return
    last = self._current
    self._current = None
    last._current.on_keyboard_interupted(self, self.get_keys_pressed())
  
  def get_listener(self) -> UIKeyboardListener:
    return self._current

  def get_keys_pressed(self) -> list:
    return [k for k in self._keys]
  
class UIKeyboardEvent:
  def __init__(self, key, key_code, last:UIKeyboardEvent=None, time:int=None):
    self._key = key
    self._key_code = key_code
    self._time = int(time * 1000) if time is not None else time
    self._last = last
  
  @property
  def key(self):
    return self._key
  
  @property
  def key_code(self):
    return self._key_code
  
  @property
  def time(self):
    return self._time
  
  @property
  def pressed(self) -> bool:
    return self.is_pressed_event()
  
  def __eq__(self, key:UIKeyboardEvent) -> bool:
    return self._key == key._key and self._key_code == key._key_code
  
  def __str__(self) -> str:
    event_type = "Pressed" if self.is_pressed_event() else "Released"
    return f"Keyboard Event: {event_type} '{self._key}' ({self._key_code})"
  
  def is_pressed_event(self) -> bool:
    return self._last is not None

class UIKeyboardListener(ABC):
  def on_key_pressed(self, keyboard:UIKeyboard, key:UIKeyboardEvent) -> bool:
    pass

  def on_key_held(self, keyboard:UIKeyboard, key:UIKeyboardEvent) -> bool:
    pass

  def on_key_released(self, keyboard:UIKeyboard, key:UIKeyboardEvent) -> bool:
    pass

  def on_keyboard_started(self, keyboard:UIKeyboard) -> bool:
    pass

  def on_keyboard_ended(self, keyboard:UIKeyboard) -> bool:
    pass

  def on_keyboard_interupted(self, keyboard:UIKeyboard, keys, cause=None) -> bool:
    pass

class UISearchCriteria(ABC):
  def filter(self, element:UIElement) -> bool:
    raise NotImplementedError()

class _PointInteractableSearchCriteria(UISearchCriteria):
  def __init__(self, x, y, excluded_objects=None, excluded_classes=None, included_classes=None):
    self.x = x
    self.y = y
    self.excluded_objects = excluded_objects or []
    self.excluded_classes = excluded_classes or []
    self.included_classes = included_classes or []

    # Convert single values to lists
    if not isinstance(self.excluded_objects, (list, tuple)):
      self.excluded_objects = [self.excluded_objects]
    if not isinstance(self.excluded_classes, (list, tuple)):
      self.excluded_classes = [self.excluded_classes]
    if not isinstance(self.included_classes, (list, tuple)):
      self.included_classes = [self.included_classes]
      
  def filter(self, element:UIElement):
    x, y = element.get_reversed_global_matrix() @ (self.x, self.y)
    #if element.uses_own_graphics or isinstance(element, UIRoot):
    if not element.is_local_point_within_rectangle(x, y):
      return False
    # Check if element is interactable at the given point
    if not element.is_point_interactable(x, y):
      return False
    # Check if element is in the list of excluded objects
    if element in self.excluded_objects:
      return False
    # Check if element's type is in the list of excluded classes
    if self.excluded_classes:
      if any(isinstance(element, excluded_class) for excluded_class in self.excluded_classes):
        return False
    # If included classes are specified, check if element's type is in the list
    if self.included_classes:
      if not any(isinstance(element, included_class) for included_class in self.included_classes):
        return False
    # If all checks pass, include this UIElement
    return True