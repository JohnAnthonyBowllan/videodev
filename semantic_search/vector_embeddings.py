from manim import *
import numpy as np
from manim.utils.color import color_to_rgb

use_distance = ValueTracker(0)  # 0 = cos, 1 = 1 - cos

class UnitCircleCommonAngles(Scene):

    def construct(self):
        # constants
        radius = 2
        center_shift = LEFT * 1.5
        angle_value = ValueTracker(30 * DEGREES)

        # unit circle
        circle = Circle(radius=radius, color=YELLOW)
        x_axis = Line(LEFT*2.5, RIGHT*2.5, color=GREY).move_to(DOWN*0.01)
        y_axis = Line(DOWN*2.5, UP*2.5, color=GREY).move_to(RIGHT*0.01)
        circle_group = VGroup(circle, x_axis, y_axis).move_to(ORIGIN)

        # vectors
        reference_vector = Arrow(
            start=ORIGIN + center_shift,
            end=radius * np.array([np.cos(0), np.sin(0), 0]) + center_shift,
            buff=0,
            color=YELLOW,
        )
        vector = always_redraw(lambda: Arrow(
            start=ORIGIN + center_shift,
            end=radius * np.array([np.cos(angle_value.get_value()), np.sin(angle_value.get_value()), 0]) + center_shift,
            buff=0,
            color=YELLOW,
        ))

        # angles and projections
        arc = always_redraw(lambda: Arc(
            radius=0.4,
            start_angle=0,
            angle=angle_value.get_value(),
            arc_center=ORIGIN + center_shift,
            color=YELLOW
        ))
        shadow_projection = always_redraw(lambda: DashedLine(
            start=radius * np.array([
                np.cos(angle_value.get_value()),
                np.sin(angle_value.get_value()),
                0]) + center_shift,
            end=[radius * np.cos(angle_value.get_value()), 0, 0] + center_shift,
            color=WHITE
        ))
        cos_brace = always_redraw(lambda: BraceBetweenPoints(
            ORIGIN + center_shift + DOWN * 0.02,
            [radius * np.cos(angle_value.get_value()), 0, 0] + center_shift + DOWN * 0.02,
            direction=DOWN,
            color=BLUE
        ))
        cos_brace_label = always_redraw(
            lambda: MathTex(f"{np.cos(angle_value.get_value()):.2f}", color=BLUE)\
                .scale(0.7)\
                .next_to(cos_brace, DOWN, buff=0.1)\
                .move_to(cos_brace.get_center() + DOWN * 0.3)
        )

        # cosine tracker bar and ticks
        cos_bar = Line(ORIGIN + RIGHT*3, ORIGIN + UP*3 + RIGHT*3, color=WHITE).shift(DOWN * 2)
        baseline = Line(
            start=np.array([-2, 0, 0]) + center_shift,
            end=np.array([2, 0, 0]) + center_shift,
            color=GREY
        )
        top_tick = Line(cos_bar.get_end() + LEFT*0.1, cos_bar.get_end() + RIGHT*0.1, color=WHITE)
        bottom_tick = Line(cos_bar.get_start() + LEFT*0.1, cos_bar.get_start() + RIGHT*0.1, color=WHITE)

        # tick labels
        minus_one_label = MathTex("-1").scale(0.6).next_to(cos_bar.get_start(), LEFT, buff=0.15)
        plus_one_label = MathTex("1").scale(0.6).next_to(cos_bar.get_end(), LEFT, buff=0.15)
        zero_label = MathTex("0").scale(0.6).next_to(cos_bar.get_start(), LEFT, buff=0.15)
        plus_two_label = MathTex("2").scale(0.6).next_to(cos_bar.get_end(), LEFT, buff=0.15)

        # tracker dots
        cos_dot = Dot(radius=0.12, color=BLUE)
        cos_dist_dot = Dot(radius=0.12, color=ORANGE)
        cos_dot.add_updater(
            lambda d: d.move_to(
                cos_bar.point_from_proportion((np.cos(angle_value.get_value()) + 1) / 2)
            )
        )
        cos_dist_dot.add_updater(
            lambda d: d.move_to(
                cos_bar.point_from_proportion((1 - np.cos(angle_value.get_value())) / 2)
            )
        )

        # formulas
        cos_label_lhs = always_redraw(
            lambda: MathTex(f"\\cos({int(np.degrees(angle_value.get_value()))}^\\circ) = ") \
                .scale(0.7) \
                .next_to(cos_bar, UP)
        )
        cos_label_rhs = always_redraw(
            lambda: MathTex(f"{np.cos(angle_value.get_value()):.2f}", color=BLUE) \
                .scale(0.7) \
                .next_to(cos_label_lhs, RIGHT)
        )
        one_minus = MathTex(f"1 - ").scale(0.7).next_to(cos_label_lhs, LEFT)

        # titles
        cos_bar_text = Text("Cosine Similarity").next_to(cos_label_lhs, UP * 1.5)
        cos_dist_text = Text("Cosine Distance").next_to(cos_label_lhs, UP * 2.0)

        # animation scheme
        self.play(Create(x_axis), Create(y_axis))
        self.wait(0.5)
        self.play(Create(circle))
        self.play(circle.animate.scale(1.2), run_time=0.3)
        self.play(circle.animate.set_color(WHITE).scale(1.0/1.2), run_time=0.3)
        self.wait(1)
        self.play(circle_group.animate.shift(center_shift))
        self.play(FadeIn(baseline, run_time=0.1))
        self.play(Transform(baseline, cos_bar))
        self.play(Create(top_tick), Create(bottom_tick), Write(minus_one_label), Write(plus_one_label))
        self.wait()
        self.play(GrowArrow(vector), GrowArrow(reference_vector))
        self.wait(0.5)
        self.play(Create(arc))
        self.wait(0.5)
        self.play(GrowFromPoint(shadow_projection,vector.get_end(), run_time=1.5))
        self.wait(0.5)
        self.play(
            Create(cos_dot),
            Write(cos_label_lhs),
            Write(cos_label_rhs),
            GrowFromCenter(cos_brace),
            Write(cos_brace_label),
        )
        self.play(Write(cos_bar_text))
        self.wait()
        angles = [0, 90, 180, 60, 45, 120, 150]
        for deg in angles:
            theta = deg * DEGREES
            self.play(angle_value.animate.set_value(theta), run_time=1.5)
            self.wait()
        self.wait(2)

        cos_dot.clear_updaters()
        cos_label_rhs.clear_updaters()
        cos_label_dist_rhs = always_redraw(
            lambda: MathTex(f"{1 - np.cos(angle_value.get_value()):.2f}", color=ORANGE) \
                .scale(0.7) \
                .next_to(cos_label_lhs, RIGHT)
        )
        self.play(
            FadeOut(plus_one_label),
            FadeOut(cos_dot),
            FadeOut(minus_one_label),
            FadeOut(cos_label_rhs),
        )
        self.wait()
        self.play(Write(one_minus))
        self.wait()
        self.play(Write(cos_label_dist_rhs))
        self.wait()
        self.play(Write(zero_label), Write(plus_two_label))
        self.wait()
        self.play(FadeIn(cos_dist_dot))
        self.wait()
        self.play(Transform(cos_bar_text, cos_dist_text))
        self.wait(2)
        angles= [0, 90, 180, 60, 45, 120, 150]
        for deg in angles:
            theta = deg * DEGREES
            self.play(angle_value.animate.set_value(theta), run_time=1.5)
            self.wait()
        self.wait(2)


class WordEmbeddingScene(Scene):
    @staticmethod
    def compute_cosine_distance(theta_a, theta_b):
        angle_diff = abs(theta_b - theta_a)
        cos_sim = np.cos(angle_diff)
        return 1 - cos_sim

    def construct(self):
        origin = ORIGIN
        vec_len = 3
        label_offset = 0.75

        def get_label_position(vec, offset=label_offset):
            direction = vec.get_end() / np.linalg.norm(vec.get_end())
            return vec.get_end() + offset * direction

        # Angle trackers
        angle_a = ValueTracker(0)
        angle_b = ValueTracker(PI / 3)

        # Vectors (white)
        vec_a = always_redraw(lambda: Arrow(
            start=origin,
            end=vec_len * np.array([
                np.cos(angle_a.get_value()),
                np.sin(angle_a.get_value()),
                0]),
            buff=0,
            stroke_width=6,
            tip_shape=StealthTip,
            color=WHITE
        ))

        vec_b = always_redraw(lambda: Arrow(
            start=origin,
            end=vec_len * np.array([
                np.cos(angle_b.get_value()),
                np.sin(angle_b.get_value()),
                0]),
            buff=0,
            stroke_width=6,
            tip_shape=StealthTip,
            color=WHITE
        ))

        word_a = Text("king", font_size=32).move_to(get_label_position(vec_a))
        word_b = Text("queen", font_size=32).move_to(get_label_position(vec_b))

        # Blend between two colors based on distance
        def blend_colors(color1, color2, t):
            """Blend two Manim colors based on t in [0,1]."""
            rgb1 = np.array(color_to_rgb(color1))
            rgb2 = np.array(color_to_rgb(color2))
            blended = (1 - t) * rgb1 + t * rgb2
            return rgb_to_color(tuple(blended))

        def get_arc():
            theta_a = angle_a.get_value()
            theta_b = angle_b.get_value()
            ###
            angle_diff = (theta_b - theta_a) % TAU
            use_other_angle = angle_diff > PI
            ###

            dist = 1 - np.cos(theta_b - theta_a)
            t = np.clip(dist / 1.5, 0, 1)

            color = blend_colors(YELLOW_B, ORANGE, t)
            stroke_width = 12 * (1 - t) + 4 * t  # thick (12) when close, thin (4) when far

            return Angle(
                vec_a, vec_b,
                radius=0.8,
                other_angle=use_other_angle,
                color=color,
                stroke_width=stroke_width
            )

        arc = always_redraw(get_arc)

        # Cosine distance display
        cosine_distance = DecimalNumber(
            number=0.0,
            num_decimal_places=2,
            color=YELLOW
        ).to_corner(UR)

        label = Text("Cosine Dist:", font_size=24).next_to(cosine_distance, LEFT)

        def update_distance(mob):
            theta_a = angle_a.get_value()
            theta_b = angle_b.get_value()
            distance = 1 - np.cos(theta_b - theta_a)
            mob.set_value(distance)

        cosine_distance.add_updater(update_distance)

        # Add everything
        # self.add(vec_a, vec_b, arc, cosine_distance, label)
        # self.play(Create(NumberPlane()))
        self.play(Create(vec_a, run_time=1.5))
        self.play(Write(word_a))
        self.play(Create(vec_b))
        self.play(Write(word_b))
        self.play(Create(arc))
        self.play(Write(label))
        self.play(Write(cosine_distance))

        # Define fixed transitions
        transitions = [
            (PI / 8, -PI / 10, "dog", "cat"),
            (-PI / 6, PI / 5, "fish", "banana"),
            (PI / 4, PI / 3, "car", "warrior"),
            (-PI / 4, PI / 3, "basketball", "wanda"),
        ]

        for da, db, label_a, label_b in transitions:
            # new_word_a = Text(label_a, font_size=32).move_to(vec_a.get_end() + 0.3 * UP)
            # new_word_b = Text(label_b, font_size=32).move_to(vec_b.get_end() + 0.3 * UP)

            new_word_a = Text(label_a, font_size=32).move_to(get_label_position(vec_a))
            new_word_b = Text(label_b, font_size=32).move_to(get_label_position(vec_b))

            # new_word_a.add_updater(lambda m: m.move_to(vec_a.get_end() + 0.3 * UP))
            # new_word_b.add_updater(lambda m: m.move_to(vec_b.get_end() + 0.3 * UP))

            new_word_a.add_updater(lambda m: m.move_to(get_label_position(vec_a)))
            new_word_b.add_updater(lambda m: m.move_to(get_label_position(vec_b)))

            self.play(
                FadeOut(word_a),
                FadeOut(word_b),
                angle_a.animate.increment_value(da),
                angle_b.animate.increment_value(db),
                FadeIn(new_word_a),
                FadeIn(new_word_b),
                run_time=2,
                rate_func=smooth
            )
            word_a = new_word_a
            word_b = new_word_b
            self.wait(0.5)

        cosine_distance.remove_updater(update_distance)