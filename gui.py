from kivy.app import App
from kivy.lang import Builder
from kivy.uix.widget import Widget
from generatesample import Generator
import random.randint

class GUILoader(Widget):
    def __init__(self):
        self.sample = Generator()

    def load_new_img(self):
        images, ans = self.sample.get_image(randint(0,100))
        return images
    
    def load_new_qn(self):
        pass

class GUIApp(App):
    def build(self):
        self.title = "VQA Sampling"
        return GUILoader()


if __name__ == '__main__':
    GUIApp().run()