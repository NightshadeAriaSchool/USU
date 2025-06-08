import pygame
import threading
from Yui import Yui, YuiRoot, MouseListener, Graphics, Color
import MNIST
from MNIST import Digit, Set
from Naming import AgentName
from Learn import Classifier, TestingReport, TrainingReport

class Global:
    agents: list[Classifier] = []
    digit_set: Set = None
    
    @staticmethod
    def init():
        Global.agents = Classifier([16, 16])
        Global.digit_set = MNIST.load()

class WindowRoot(YuiRoot):
    def __init__(self, width = 800, height = 600, framerate = 60, name = 'Yui Window', is_resizable = False):
        super().__init__(width, height, framerate, name, is_resizable)
    
    def on_draw(self, graphics: Graphics):
        graphics.background(Color(0, 0, 0, 255))
    
    def on_child_destroyed(self, child, index):
        if isinstance(child, LoadingScreen):
            pass # TODO

class LoadingScreen(Yui):
    def __init__(self, parent):
        super().__init__(parent)
        self.destroy_timer = 20
        self.width = self.root.width
        self.height = self.root.height
        
        threading.Thread(target=Global.init).start()
    
    def on_draw(self, graphics: Graphics):
        graphics.text_align_x, graphics.text_align_y = 0.5, 0.5
        graphics.fill_color = Color(255, 255, 255, 255)
        graphics.text_size = 20
        
        if Global.digit_set:
            graphics.text("Loaded!", self.width / 2, self.height / 2)
            self.destroy_timer -= 1
            if self.destroy_timer == 0:
                self.destroy()
        else:
            graphics.text("Loading...", self.width / 2, self.height / 2)

root = WindowRoot(640, 480)
loading_screen = LoadingScreen(root)
root.init()