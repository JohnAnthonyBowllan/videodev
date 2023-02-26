from manim import PI, Circle, Create, FadeOut, Scene, Square, Transform


class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle

        square = Square()  # create a square
        square.rotate(PI / 4)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation
        # test comment
