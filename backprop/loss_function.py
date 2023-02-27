# from manim import PI, Circle, Create, FadeOut, Scene, Square, Transform
from manim import *


class LinearResiduals(Scene):
    def construct(self):
        coords = [(1, -7), (2, 5), (5, 11), (5, 8), (1, 1), (4, 3), (3, 1), (4, 5)]
        x_min = min([coord[0] for coord in coords]) - 3
        x_max = max([coord[0] for coord in coords]) + 3
        y_min = min([coord[1] for coord in coords]) - 3
        y_max = max([coord[1] for coord in coords]) + 3
        grid = Axes(
            x_range=[x_min, x_max, 1],
            y_range=[y_min, y_max, 1],
            x_length=9,
            y_length=9,
            tips=False,
        )
        points = VGroup(*[Dot(color=YELLOW).move_to(grid.c2p(coord[0], coord[1])) for coord in coords])
        self.add(grid, points)
        m, b = ValueTracker(397 / 151), ValueTracker(-731 / 151)
        # linear_function = grid.plot(lambda x: m.get_value()*x + b.get_value(), x_range=[x_min, x_max, 1])

        def get_linear_function():
            linear_function = Line(
                start=grid.c2p(x_min, m.get_value()*x_min + b.get_value()),
                end=grid.c2p(x_max, m.get_value()*x_max + b.get_value())
            )
            return linear_function

        def mse(y_true_arr, y_pred_arr):
            return sum([(y_true - y_pred)**2 for y_true, y_pred in zip(y_true_arr, y_pred_arr)])/len(y_true_arr)

        regression_line = get_linear_function()
        moving_line = regression_line.copy()
        self.add(moving_line)

        moving_line.add_updater(lambda l: l.become(get_linear_function()))

        residual_vertical_lines = [
            DashedLine(
                start=grid.c2p(coord[0], coord[1]),
                end=grid.c2p(coord[0], m.get_value() * coord[0] + b.get_value())
            )
            for coord in coords
        ]
        moving_residual_vertical_lines = []

        for i, res_line in enumerate(residual_vertical_lines):
            moving_res_line = res_line.copy()
            moving_res_line.add_updater(lambda l: l.become(
                    DashedLine(
                        l.start,
                        grid.c2p(grid.p2c(l.start)[0], m.get_value() * grid.p2c(l.start)[0] + b.get_value())
                    )
                )
            )
            moving_residual_vertical_lines.append(moving_res_line)
        moving_residual_vertical_lines = VGroup(*moving_residual_vertical_lines)
        loss = DecimalNumber(mse(
            y_true_arr=[coord[1] for coord in coords],
            y_pred_arr=[m.get_value() * coord[0] + b.get_value() for coord in coords]
        ), num_decimal_places=3, include_sign=True, unit=None)
        loss.add_updater(lambda d: d.set_value(
            mse(
                y_true_arr=[coord[1] for coord in coords],
                y_pred_arr=[m.get_value() * coord[0] + b.get_value() for coord in coords]
            )
        ))
        self.add(moving_residual_vertical_lines, loss.move_to(UP*3 + RIGHT *4))
        self.wait()
        self.play(m.animate.set_value(-1), b.animate.set_value(5))


        # have copies of residual lines that grow and shrink and add + inbetween (to show loss grows when residual lengths are greater

"""
(1/8) * ( (-7 - ((397/151)*1 + (-731/151)))^2 + (5 - ((397/151)*2 + (-731/151)))^2 + (11 - ((397/151)*5 + (-731/151)))^2 + (8 - ((397/151)*5 + (-731/151)))^2 + (1 - ((397/151)*1 + (-731/151)))^2 + (3 - ((397/151)*4 + (-731/151)))^2 + (1 - ((397/151)*3 + (-731/151)))^2 + (5 - ((397/151)*4 + (-731/151)))^2)

# [https://www.wolframalpha.com/input?i=z+%3D+(1%2F8)+*+(+(-7+-+(x*1+%2B+y))^2+%2B+(5+-+(x*2+%2B+y))^2+%2B+(11+-+(x*5+%2B+y))^2+%2B+(8+-+(x*5+%2B+y))^2+%2B+(1+-+(x*1+%2B+y))^2+%2B+(3+-+(x*4+%2B+y))^2+%2B+(1+-+(x*3+%2B+y))^2+%2B+(5+-+(x*4+%2B+y))^2)](https://www.wolframalpha.com/input?i=z+%3D+%281%2F8%29+*+%28+%28-7+-+%28x*1+%2B+y%29%29%5E2+%2B+%285+-+%28x*2+%2B+y%29%29%5E2+%2B+%2811+-+%28x*5+%2B+y%29%29%5E2+%2B+%288+-+%28x*5+%2B+y%29%29%5E2+%2B+%281+-+%28x*1+%2B+y%29%29%5E2+%2B+%283+-+%28x*4+%2B+y%29%29%5E2+%2B+%281+-+%28x*3+%2B+y%29%29%5E2+%2B+%285+-+%28x*4+%2B+y%29%29%5E2%29)
"""