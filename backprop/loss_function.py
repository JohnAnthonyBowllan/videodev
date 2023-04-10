# from manim import PI, Circle, Create, FadeOut, Scene, Square, Transform
from manim import *
from typing import List, Union


class LinearResiduals(Scene):
    def construct(self):
        coords = [(1, -7), (2, 5), (9, 4), (5, 8), (-2, 1), (8, 3), (3, 1), (4, 5)]
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
        m, b = ValueTracker(1/2), ValueTracker(-1)
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
        ), num_decimal_places=2, include_sign=True, unit=None)
        loss.add_updater(lambda d: d.set_value(
            mse(
                y_true_arr=[coord[1] for coord in coords],
                y_pred_arr=[m.get_value() * coord[0] + b.get_value() for coord in coords]
            )
        ))
        self.add(moving_residual_vertical_lines, loss.move_to(UP*3 + RIGHT *4))
        self.wait()
        self.play(m.animate.set_value(-1), b.animate.set_value(5), run_time=3)


class ReduceLossEquation(Scene):
    def construct(self):
        # CONFIGURE ME
        sum_term = r"(y_{i} - (m \times x_{i} + b))"
        left_alignment = LEFT*9
        scale = 0.6
        lower_bound = 1
        upper_bound = 4

        # base sum symbols
        sum_symbol = r"\sum_{{i={}}}^{{{}}}"
        sum_term = sum_term.replace("{i}", "{{{index}}}")

        # create base equation
        base_eqn_terms = [
            sum_term.format(index=lower_bound - 1) + " +",
            rf"{sum_symbol.format(lower_bound, upper_bound)} {sum_term.format(index='i')}"
        ]
        base_eqn = MathTex(*base_eqn_terms).scale(scale).align_to(left_alignment, LEFT)
        base_eqn.set_color_by_tex(sum_term.format(index=lower_bound - 1), BLACK)
        self.play(Write(base_eqn))

        # expand sum
        for i in range(lower_bound, upper_bound):
            # expanded terms for new equation
            new_eqn_terms = [
                sum_term.format(index=j) + " +"
                for j in range(lower_bound-1, i)
            ]
            new_eqn_terms_v2 = new_eqn_terms.copy()

            # final term (e.g. sigma) for new equation
            if i < upper_bound - 1:
                new_eqn_terms += [
                    rf"{sum_term.format(index=i) + ' +'} {sum_symbol.format(i + 1, upper_bound)} {sum_term.format(index='i')}"
                ]
                new_eqn_terms_v2 += [sum_term.format(index=i) + ' +', rf"{sum_symbol.format(i + 1, upper_bound)} {sum_term.format(index='i')}"]
            else:
                last_two_terms = [
                    rf"{sum_term.format(index=i) + ' + ' + sum_term.format(index=i+1)}"
                ]
                new_eqn_terms += last_two_terms
                new_eqn_terms_v2 += last_two_terms

            new_eqn = MathTex(*new_eqn_terms).scale(scale).align_to(left_alignment, LEFT)
            new_eqn.set_color_by_tex(sum_term.format(index=lower_bound - 1), BLACK)

            # transform base equation to new equation
            self.play(*[
                ReplacementTransform(
                    base_eqn.get_part_by_tex(base_eqn_terms[j]),
                    new_eqn.get_part_by_tex(new_eqn_terms[j])
                ) for j in range(lower_bound-1, i+1)
            ], run_time=0.75)

            # v2 is same as new equation except it splits the last tex into two
            new_eqn_v2 = MathTex(*new_eqn_terms_v2).scale(scale).align_to(left_alignment, LEFT)
            new_eqn_v2.set_color_by_tex(sum_term.format(index=lower_bound - 1), BLACK)
            base_eqn_terms = new_eqn_terms_v2
            base_eqn = new_eqn_v2
            self.play(FadeIn(new_eqn_v2), FadeOut(new_eqn), run_time=0.01)


class ThreeDLossSurfaceConstruction(ThreeDScene):
    def construct(self):
        pass
