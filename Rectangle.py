class Rectangle:

    # array is consisted of left,top, width, height
    def __init__(self, array):
        self.left = array[0]
        self.top = array[1]
        self.right = self.left + array[2]
        self.bottom = self.top + array[3]

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def get_top(self):
        return self.top

    def get_bottom(self):
        return self.bottom

    def contains_point_on_x_axcis(self, point):
        return self.left <= point <= self.right

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right

    def set_top(self, top):
        self.top = top

    def set_bottom(self, bottom):
        self.bottom = bottom