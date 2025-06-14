import pygame
import threading
from Yui import Yui, YuiRoot, Graphics, Color, MouseListener, MouseEvent, Stack, Button, TextField, Slider, Vector2D
import MNIST
from MNIST import Digit, Set
from Naming import AgentName
from Learn import Classifier, TestingReport, TrainingReport
import torch
import numpy as np
import re
import random

# Kitty hates this T_T
# TODO:
# |    Add Classifier tree view and ability to select Global.selected_agent
# |    Add the whole canvas bullshit
# |    Go nap

class Global:
    USER_GRAPHICS_SCALE: int = 10
    
    initialized: bool = False
    dataset: Set = None      
    agents: list[Classifier] = []
    selected_agent: Classifier = None # Initializes in init()
    selected_train_epochs: int = 1
    learning_rate: int = 0.01
    user_canvas: Graphics = None
    user_digit: Digit = None
    prediction: list[float] = None
    batch_size: int = -1
    batch_seed: int = 69
          
    @staticmethod      
    def init():      
        Global.dataset = MNIST.load(split_train_and_test=False, max_size=5000)
        Global.agents.append(Classifier(layer_sizes=[128, 64]))
        Global.selected_agent = Global.agents[0]
        Global.selected_train_epochs = 1
        Global.learning_rate = 0.01
        Global.user_canvas = Graphics(28 * Global.USER_GRAPHICS_SCALE, 28 * Global.USER_GRAPHICS_SCALE)
        Global.user_digit = MNIST.Digit.from_graphics(Global.user_canvas)
        Global.prediction = Global.predict()
        Global.batch_size = len(Global.dataset)
        Global.batch_seed = 69
        Global.initialized = True
    @staticmethod
    def train(epochs: int):
        for _ in range(epochs):
            forking = Global.selected_agent.trained
            data = Global.dataset.random(size=Global.batch_size, seed=Global.batch_seed)
            Global.batch_seed = random.Random(Global.batch_seed).randint(0, 2**32)
            new_agent = Global.selected_agent.train(data, learning_rate = Global.learning_rate)
            if forking:
                Global.agents.append(new_agent)                      
            else:                        
                idx = Global.agents.index(Global.selected_agent)
                Global.agents[idx] = new_agent
            Global.selected_agent = new_agent
    @staticmethod
    def predict() -> torch.Tensor:
        Global.user_digit = MNIST.Digit.from_graphics(Global.user_canvas)
        # If digit is not a torch.Tensor, convert here
        if not isinstance(Global.user_digit.tensor, torch.Tensor):
            digit_tensor = torch.tensor(Global.user_digit.tensor, dtype=torch.float32)
        else:
            digit_tensor = Global.user_digit.tensor.float()
        # Predict using the selected agent
        output = Global.selected_agent.predict(Digit(digit_tensor))
        Global.prediction = output
        return output


class Main(YuiRoot):      
    def __init__(self, width=800, height=600, name='NN Playground', framerate=60):    
        super().__init__(width=width, height=height, name=name, framerate=framerate)
        self.auto_background = Color(0, 0, 0, 255)
        self.auto_draw_bounds = False
        LoadingScreen(parent=self)      
    
    def on_draw(self, graphics: Graphics):
        # graphics.background(Color(0, 0, 0, 255))
        
        graphics.fill_color = Color(255, 255, 255, 255)
        graphics.text_size = 15
        graphics.text_align_x, graphics.text_align_y = 0, 1
        graphics.text(f"Pressed Yui: {self.mouse.pressed}", 0, self.height - 30)
        if self.mouse.current:
            graphics.text(f"Buttons Down: {bin(self.mouse.current.down)}", 0, self.height - 15)
            graphics.text(f"Mouse Event: {bin(self.mouse.current.event)}", 0, self.height)
    def on_child_destroyed(self, child: Yui, index: int):
        if isinstance(child, LoadingScreen):
            ActualProgram(parent=self)
      
class LoadingScreen(Yui):      
    def __init__(self, parent=Yui):      
        super().__init__(parent=parent)      
        self.ticks_to_destroy = 20      
        threading.Thread(target=Global.init).start()
    def on_draw(self, graphics: Graphics):      
        self.width, self.height = self.root.width, self.root.height      
              
        graphics.fill_color = Color(255, 255, 255, 255)      
        graphics.text_size = 40
        graphics.text_align_x, graphics.text_align_y = 0, 1      
              
        if Global.initialized:      
            self.ticks_to_destroy -= 1      
            if self.ticks_to_destroy == 0:
                self.destroy()      
                return      
            graphics.text("Loaded!", self.width * 3 / 8, self.height / 2)      
        else:      
            graphics.text("Loading...", self.width * 3 / 8, self.height / 2)
            
            graphics.text_size = 25
            graphics.text_align_x, graphics.text_align_y = 1, 1
            graphics.text(f"{MNIST.progress} / {MNIST.total}", self.width * 5 / 8, self.height / 2)
            
            graphics.stroke_color = Color(255, 255, 255, 255)
            graphics.stroke_width = 1
            graphics.line(self.width * 3 / 8, self.height / 2, self.width * 3 / 8 + self.width / 4 * (MNIST.progress / MNIST.total), self.height / 2)
            graphics.point(self.width * 3 / 8, self.height / 2)
            graphics.point(self.width * 5 / 8, self.height / 2)


class ActualProgram(Stack):      
    def __init__(self, parent: Yui):
        super().__init__(parent=parent, is_vertical=False)
        self.width, self.height = self.root.width, self.root.height
        self.stack_margin = self.width / 80
        
        class Canvas(Yui, MouseListener):
            def __init__(self, parent):
                super().__init__(parent)
            
            def on_draw(self, graphics):
                scale = min(self.width / Global.user_canvas.width, self.height / Global.user_canvas.height)
                
                graphics.no_fill()
                graphics.stroke_width = 2
                graphics.stroke_color = Color(255, 255, 255, 255)
                graphics.rect_mode = 'center'
                graphics.rectangle(self.width * 0.5, self.height * 0.5, Global.user_canvas.width * scale, Global.user_canvas.height * scale)
                
                graphics.image_mode = 'center'
                graphics.image(Global.user_canvas, self.width * 0.5, self.height * 0.5, Global.user_canvas.width * scale, Global.user_canvas.height * scale)
            
            def on_mouse_event(self, event):
                if not event.any_button_down and not event.is_released_event:
                    return
                
                scale = min(self.width / Global.user_canvas.width, self.height / Global.user_canvas.height)
                width = Global.user_canvas.width * scale
                height = Global.user_canvas.height * scale
                left = (self.width - width) * 0.5
                top = (self.height - height) * 0.5
                
                last = (event.mouse.last.point - Vector2D(left, top)) / scale
                point = (event.point - Vector2D(left, top)) / scale
                
                Global.user_canvas.ellipse_mode = 'center'
                Global.user_canvas.no_stroke()
                Global.user_canvas.fill_color = Color(255, 255, 255, 255)
                for i in range(21):
                    lerp = point + (last - point) * i / 20
                    Global.user_canvas.ellipse(lerp.x, lerp.y, Global.USER_GRAPHICS_SCALE * 3, Global.USER_GRAPHICS_SCALE * 3)
                
                if event.is_released_event:
                    Global.predict()

        class PredictionYui(Stack):
            def __init__(self, parent):
                super().__init__(parent=parent, is_vertical=False)
                self.stack_margin = 10

            def on_draw(self, graphics: Graphics):
                prediction = Global.prediction
                if prediction is None:
                    return
                values = prediction.detach().cpu().numpy()
                values = values.flatten()
                max_val = max(max(values), 1)
                
                graphics.no_fill()
                graphics.stroke_width = 1
                graphics.stroke_color = Color(255, 255, 255, 255)
                graphics.rect_mode = 'corner'
                graphics.rectangle(0, 0, self.height, self.height)
                graphics.image_mode = 'corner'
                graphics.image(Global.user_digit.graphics, 0, 0, self.height, self.height, smooth=False)
                
                for i, val in enumerate(values):
                    w = (self.width - self.height) / 10
                    x = self.height + i * w
                    h = self.height
                    # Draw digit
                    graphics.fill_color = Color(255, 255, 255, 255)
                    graphics.text_size = int(h * 0.5)
                    graphics.text_align_x = 0.5
                    graphics.text_align_y = 0.7
                    graphics.text(str(i), x + w / 2, h * 0.5)
                    # Draw confidence
                    graphics.text_size = int(h * 0.18)
                    graphics.text_align_y = 1
                    graphics.text(f"{float(val):.2f}", x + w / 2, h * 0.95)
                    # Draw bar
                    bar_h = h * 0.25 * (val / max_val if max_val > 0 else 0)
                    bar_x = x + w - 5
                    bar_y = h * 0.5
                    r = int(max(0, min(255, 127 - 127 * val)))
                    g = 0
                    b = int(max(0, min(255, 127 + 127 * val)))
                    graphics.fill_color = Color(63, 255, 63, 255) if val == max_val else Color(r, g, b, 255)
                    graphics.rect_mode = 'corner'
                    graphics.rectangle(bar_x, bar_y, 8, bar_h)
        
        class InfoYui(Yui):
            def __init__(self, parent):
                super().__init__(parent)
            
            def on_draw(self, graphics):
                graphics.fill_color = Color(255, 255, 255, 255)
                graphics.text_align_x = 1
                graphics.text_align_y = 0
                
                graphics.text_size = int(self.height / 2)
                graphics.text(f"{Global.selected_agent.name.persona_name}", self.width, 0)
                graphics.text_size = int(self.height / 6)
                graphics.text(f"Epoch: {Global.selected_agent.name.epoch}", self.width, self.height * 3 / 6)
                forked_by = Global.selected_agent.forked_by
                if forked_by:
                    graphics.text(f"Forked by: {forked_by.name.persona_name}", self.width, self.height * 4 / 6)
                    graphics.text(f"Forked at: {forked_by.name.epoch}", self.width, self.height * 5 / 6)

        class AgentTreeView(Yui, MouseListener):
            def __init__(self, parent):
                super().__init__(parent)
                self.uses_graphics = True
                self.centered_index = 0
            
            @staticmethod
            def build_agent_tree():
                class AgentNode:
                    def __init__(self, ref):
                        self.ref = ref                 # Classifier
                        self.forked_by = ref.forked_by
                        self.id = self.ref.name.id
                        self.parent = None
                        self.children = []
                        self.order = -1               # Vertical drawing row

                nodes_by_ref = [AgentNode(agent) for agent in Global.agents]
                
                # Sort nodes: those with no forked_by first, then by forked_by.name.epoch
                nodes_by_ref.sort(key=lambda node: (-2**30 if node.forked_by is None else - node.forked_by.name.epoch))

                # Connect parents and children
                for child in nodes_by_ref:
                    for parent in nodes_by_ref:
                        if child.forked_by and child.forked_by.name.id == parent.ref.name.id:
                            child.parent = parent
                            parent.children.append(child)
                
                nodes_by_ref = [node for node in nodes_by_ref if not node.parent]
                
                # Flatten and assign order
                def flatten(node, result=[]):
                    result.append(node)
                    for child in node.children:
                        flatten(child, result)
                    return result
                
                result = []
                for node in nodes_by_ref:
                    flatten(node, result)
                
                for i, node in enumerate(result):
                    node.order = i
                return result

            def on_draw(self, graphics: Graphics):
                if self.uses_graphics:
                    graphics.background(Color(0, 0, 0, 255))
                nodes = AgentTreeView.build_agent_tree()
                max_epoch = max([node.ref.name.epoch for node in nodes])
                
                spacing_x = self.width / max(max_epoch, 1)  # Horizontal distance per epoch
                spacing_y = self.height / len(nodes)  # Vertical distance per order

                for node in nodes:
                    xe = node.ref.name.epoch * spacing_x
                    ye = (node.order + 0.5) * spacing_y
                    xo = (node.forked_by.name.epoch) * spacing_x if node.forked_by else 0
                    yo = (node.order + 0.5) * spacing_y
                    
                    graphics.stroke_color = Color.from_hsb(node.ref.name.color / 3, 255, 255, 255)
                    graphics.stroke_width = 1
                    graphics.line(xo, yo, xe, ye)
                    if node.parent:
                        xf = (node.forked_by.name.epoch - 1) * spacing_x
                        yf = (node.parent.order + 0.5) * spacing_y
                        graphics.line(xo, yo, xf, yf)
                    
                    graphics.fill_color = Color(255, 255, 255, 255)
                    graphics.text_size = int(min(20, spacing_y))
                    graphics.text_align_x = 1
                    graphics.text_align_y = 1
                    graphics.text(node.ref.name.persona_name, self.width, yo)
                        
                if self.root.mouse.current:
                    local = self.root.mouse.current.to_local(self)
                    if self.is_in_local_bounds(local.point):
                        selected_x = (int(local.point.x / spacing_x) + 1) * spacing_x - 1
                        selected_y = int(local.point.y / spacing_y) * spacing_y
                        graphics.stroke_color = Color(255, 255, 255, 255)
                        graphics.line(selected_x, 0, selected_x, self.height)
                        graphics.line(0, selected_y, self.width, selected_y)
                        graphics.line(0, selected_y + spacing_y, self.width, selected_y + spacing_y)
            
            def on_mouse_event(self, event: MouseEvent):
                nodes = AgentTreeView.build_agent_tree()
                max_epoch = max([node.ref.name.epoch for node in nodes])
                
                spacing_x = self.width / max(max_epoch, 1)  # Horizontal distance per epoch
                spacing_y = self.height / len(nodes)  # Vertical distance per order
                
                if event.any_button_down:
                    selected_index = int(event.point.y / spacing_y)
                    agent = nodes[selected_index].ref
                    selected_epoch = min(agent.name.epoch, max(int(event.point.x / spacing_x), 0)) + 1
                    Global.selected_agent = agent.get_epoch(int(selected_epoch))
                    Global.predict()

        class EpochValue(Yui):
            def __init__(self, parent):                        
                super().__init__(parent=parent)                        
                self._value = 1
                
                class _NumberField(TextField):
                    def on_text_changed(self, previous: str):                        
                        if not all([ch in "0123456789" for ch in self.input_text]):                        
                            self.input_text = previous                        
                        elif self.input_text and self.input_text[0] == "0":                        
                            self.input_text = previous                        
                    def on_text_finalized(self, previous: str, interupted: bool):                        
                        if not self.input_text:                        
                            self.input_text = "1"                        
                        self.parent.value = int(self.input_text)
                    
                class _ValueButton(Button):
                    def __init__(self, parent: Yui, text: str, value: int):                        
                        super().__init__(parent, text)                        
                        self.value = value                 
                    def on_click(self):                        
                        self.parent.value += self.value                        
                             
                self.number_field = _NumberField(parent=self)                        
                self.number_field.text_color = Color(255, 255, 255, 255)
                self.number_field.input_text = "1"
                self.decrement_button = _ValueButton(parent=self, text="v", value=-1)                        
                self.increment_button = _ValueButton(parent=self, text="^", value=1)                        
                                
            @property                        
            def value(self) -> int:                        
                return self._value                        
            @value.setter                        
            def value(self, value: int):                        
                old = self._value                      
                value = max(1, value)                        
                if self._value != value:                        
                    self._value = value                        
                if int(self.number_field.input_text) != self._value:                        
                    self.number_field.input_text = str(self.value)                        
                self.on_value_changed(old=old)                      
                                    
            def on_draw(self, graphics: Graphics):                        
                button_size = self.height / 2      
                    
                self.number_field.width, self.number_field.height = self.width - button_size, self.height      
                self.number_field.x, self.number_field.y = 0, 0      
                    
                self.increment_button.width, self.increment_button.height = button_size, button_size      
                self.increment_button.x, self.increment_button.y = self.width - button_size, 0      
                    
                self.decrement_button.width, self.decrement_button.height = button_size, button_size      
                self.decrement_button.x, self.decrement_button.y = self.width - button_size, button_size      
            def on_child_destroyed(self, child: Yui, index: int):                        
                self.destroy() # Destroys subtree
            
            def on_value_changed(self, old: int):
                Global.selected_train_epochs = self.value  
        
        left = Stack(self, is_vertical=True)
        left.stack_align = 0.5
        left.stack_margin = 10
        
        canvas = Canvas(left)
        canvas.width, canvas.height = self.width / 2 - self.stack_margin / 2, self.height / 2
        
        tool_stack = Stack(left, is_vertical=False)
        
        clear_canvas_button = Button(tool_stack, label="Clear Canvas")
        clear_canvas_button.width, clear_canvas_button.height = self.width / 4 - self.height / 50 - self.stack_margin / 2, self.height / 12
        def clear_canvas_on_draw(self, graphics: Graphics):
            self.input_text = f"Clear Canvas"
            Button.on_draw(self, graphics)
        def clear_canvas_on_click(self):  
            Global.user_canvas = Graphics(28 * Global.USER_GRAPHICS_SCALE, 28 * Global.USER_GRAPHICS_SCALE)
            Global.predict()
        clear_canvas_button.on_draw = clear_canvas_on_draw.__get__(clear_canvas_button, Button)
        clear_canvas_button.on_click = clear_canvas_on_click.__get__(clear_canvas_button, Button)
        
        prediction_yui = PredictionYui(left)
        prediction_yui.width, prediction_yui.height = self.width / 2 - self.stack_margin / 2, self.height / 6
        
        right = Stack(self, is_vertical=True)
        right.stack_align = 0.5
        right.stack_margin = 10
        
        info_yui = InfoYui(right)
        info_yui.width, info_yui.height = self.width / 2 - self.stack_margin / 2, self.height / 6
        
        tree_view = AgentTreeView(right)
        tree_view.width, tree_view.height = self.width / 2 - self.stack_margin / 2, self.height / 6
        
        train_stack = Stack(right, is_vertical=False)
        train_stack.stack_margin = self.height / 50 + self.stack_margin / 2
        
        epoch_value = EpochValue(train_stack)  
        epoch_value.width, epoch_value.height = self.width / 4 - self.height / 50 - self.stack_margin / 2, self.height / 12
        
        train_button = Button(train_stack)
        train_button.width, train_button.height = self.width / 4 - self.height / 50 - self.stack_margin / 2, self.height / 12
        def train_button_on_draw(self: Button, graphics: Graphics):
            self.label = f"Train {Global.selected_train_epochs} epoch{'' if Global.selected_train_epochs == 1 else 's'}"
            Button.on_draw(self, graphics)
        def train_button_on_click(self: Button):
            Global.train(Global.selected_train_epochs)
            Global.predict()
        train_button.on_draw = train_button_on_draw.__get__(train_button, Button)
        train_button.on_click = train_button_on_click.__get__(train_button, Button)
        
        batch_size_stack = Stack(right, is_vertical=False)
        batch_size_stack.stack_margin = self.height / 50 + self.stack_margin / 2
        batch_size_stack.stack_align = 0.5
        
        batch_name = TextField(batch_size_stack, is_editable=False)
        batch_name.width, batch_name.height = self.width / 8 - self.height / 50 - self.stack_margin / 2, self.height / 18
        batch_name.text_color = Color(255, 255, 255, 255)
        batch_name.default_text = "Batch size"
        
        batch_slider = Slider(batch_size_stack)  
        batch_slider.width, batch_slider.height = self.width / 4 - self.height / 50 - self.stack_margin / 2, self.height / 24
        batch_slider.minimum, batch_slider.maximum = 0, len(Global.dataset)
        batch_slider.steps = 15
        def batch_slider_on_value_changed(self: Slider, old: float):
            text_field: TextField = self.parent.children[2]
            amount = int(max(2, min(len(Global.dataset), self.value)))
            text_field.default_text = f"{amount}"
            Global.batch_size = amount
        batch_slider.on_value_changed = batch_slider_on_value_changed.__get__(batch_slider, Slider)
        
        batch_count = TextField(batch_size_stack, is_editable=False)
        batch_count.width, batch_count.height = self.width / 8 - self.height / 50 - self.stack_margin / 2, self.height / 18
        batch_count.text_color = Color(255, 255, 255, 255)
        
        batch_slider.normalized_value = 1
        
        seed_size_stack = Stack(right, is_vertical=False)
        seed_size_stack.stack_margin = self.height / 50 + self.stack_margin / 2
        seed_size_stack.stack_align = 0.5
        
        seed_name = TextField(seed_size_stack, is_editable=False)
        seed_name.width, seed_name.height = self.width / 8 - self.height / 50 - self.stack_margin / 2, self.height / 18
        seed_name.text_color = Color(255, 255, 255, 255)
        seed_name.default_text = "Seed:"
        
        seed_count = TextField(seed_size_stack, is_editable=True)
        seed_count.width, seed_count.height = self.width * 3 / 8 - self.height / 50 - self.stack_margin / 2, self.height / 18
        seed_count.text_color = Color(255, 255, 255, 255)
        seed_count.input_text = str(Global.batch_seed)
        def seed_on_draw(self: TextField, graphics: Graphics):
            if not self.is_focused and int(self.input_text) != Global.batch_seed:
                self.input_text = str(Global.batch_seed)
            TextField.on_draw(self, graphics)
        def seed_on_text_changed(self: TextField, old: str):
            if (self.input_text == "" or self.input_text == "-") and self.is_focused:
                return
            # Allow "-" as the start of a negative number, but not just "-"
            if re.fullmatch(r"-?\d*", self.input_text):
                return
            self.input_text = old
        def seed_on_text_finalized(self: TextField, previous: str, interupted: bool):
            try:
                Global.batch_seed = int(self.input_text)
            except Exception:
                self.input_text = previous
        seed_count.on_draw = seed_on_draw.__get__(seed_count, TextField)
        seed_count.on_text_changed = seed_on_text_changed.__get__(seed_count, TextField)
        seed_count.on_text_finalized = seed_on_text_finalized.__get__(seed_count, TextField)
        

if __name__ == '__main__':
    Main(1080, 585).init()
