from manim import *
import itertools as it


class MatrixScene(Scene):

    def construct(self):

        n_rows = 5
        n_cols = 4
        v_buff = 0.9
        h_buff = 1.1
        max_radius = 0.2
        min_radius = 0.02
        n_possible_radius_options = 100
        n_transitions = 5

        all_possible_dot_radi = np.linspace(min_radius, max_radius, n_possible_radius_options)
        matrix_metadata_lst = [
            [
                [
                    (
                        np.random.choice(all_possible_dot_radi),
                        GREEN if np.random.uniform() > 0.5 else RED
                    )
                    for _ in range(n_cols)
                ]
                for _ in range(n_rows)
            ]
            for _ in range(1 + n_transitions)
        ]
        source_numbers_matrix = []
        for _ in range(n_rows):
            row = []
            for _ in range(n_cols):
                row.append(ValueTracker(np.random.normal()))
            source_numbers_matrix.append(row)
        source_decimals_matrix = [
            [
                DecimalNumber(
                    source_numbers_matrix[i][j].get_value(), include_sign=True
                ).move_to(i * v_buff * DOWN + j * h_buff * RIGHT, DR).scale(0.7)
                for j in range(n_cols)
            ]
            for i in range(n_rows)
        ]

        for i in range(n_rows):
            for j in range(n_cols):
                source_decimals_matrix[i][j].add_updater(
                    lambda d, i=i, j=j: d.set_value(source_numbers_matrix[i][j].get_value())
                )
        source_matrix = [
            [
                Dot(
                    radius=matrix_metadata_lst[0][i][j][0],
                    color=matrix_metadata_lst[0][i][j][1]
                ).move_to(i * v_buff * DOWN + j * h_buff * RIGHT, DR)
                for j in range(n_cols)
            ]
            for i in range(n_rows)
        ]
        source_numbers_matrix_mobj = VGroup(*it.chain(*source_decimals_matrix)).move_to(ORIGIN)
        source_matrix_mobj = VGroup(*it.chain(*source_matrix)).move_to(ORIGIN)

        empty_tex_array = "".join(
            [
                r"\begin{array}{c}",
                *4 * [r"\quad \\"],
                r"\end{array}",
            ]
        )
        tex_left = "".join(
            [
                r"\left" + "[",
                empty_tex_array,
                r"\right.",
            ]
        )
        tex_right = "".join(
            [
                r"\left.",
                empty_tex_array,
                r"\right" + "]",
            ]
        )
        l_bracket = MathTex(tex_left).next_to(source_matrix_mobj, LEFT)
        r_bracket = MathTex(tex_right).next_to(source_matrix_mobj, RIGHT)

        l_bracket.stretch_to_fit_height(source_matrix_mobj.height + 0.5)
        r_bracket.stretch_to_fit_height(source_matrix_mobj.height + 0.5)
        # self.add(l_bracket, r_bracket, source_matrix_mobj)
        self.add(l_bracket, r_bracket, source_numbers_matrix_mobj)
        self.wait()

        for k in range(n_transitions):
            value_transitions = []
            for i in range(n_rows):
                for j in range(n_cols):
                    rand = np.random.normal()
                    value_transitions.append(source_numbers_matrix[i][j].animate.set_value(rand))
            self.play(*value_transitions, run_time=1.5)
        self.play(
            FadeOut(source_numbers_matrix_mobj),
            FadeIn(source_matrix_mobj)
        )
        self.wait()
        for k in range(n_transitions):
            source = matrix_metadata_lst[k]
            dest = matrix_metadata_lst[k+1]
            dot_transitions_same_color_pt1 = []
            dot_transitions_same_color_pt2 = []
            dot_transitions_diff_color_pt1 = []
            dot_transitions_diff_color_pt2 = []
            for i in range(n_rows):
                for j in range(n_cols):
                    if source[i][j][1] == dest[i][j][1]:
                        dot_transitions_same_color_pt1.append(
                            Transform(
                                source_matrix[i][j],
                                Dot(
                                    radius=(source[i][j][0] + dest[i][j][0])/2,
                                    color=dest[i][j][1]
                                ).move_to(source_matrix[i][j].get_center()),
                                run_time=1
                            )
                        )
                        dot_transitions_same_color_pt2.append(
                            Transform(
                                source_matrix[i][j],
                                Dot(
                                    radius=dest[i][j][0],
                                    color=dest[i][j][1]
                                ).move_to(source_matrix[i][j].get_center()),
                                run_time=1
                            )
                        )
                    else:
                        dot_transitions_diff_color_pt1.append(
                            Transform(
                                source_matrix[i][j],
                                Dot(
                                    radius=min_radius,
                                    color=source[i][j][1]
                                ).move_to(source_matrix[i][j].get_center()),
                                run_time=1,
                            )
                        )
                        dot_transitions_diff_color_pt2.append(
                            Transform(
                                source_matrix[i][j],
                                Dot(
                                    radius=dest[i][j][0],
                                    color=dest[i][j][1]
                                ).move_to(source_matrix[i][j].get_center()),
                                run_time=1
                            )
                        )
            self.play(*dot_transitions_diff_color_pt1 + dot_transitions_same_color_pt1)
            self.play(*dot_transitions_diff_color_pt2 + dot_transitions_same_color_pt2)


class WordEmbeddingScene(MovingCameraScene):

    def construct(self):
        def compute_cosine_similarity(theta_a, theta_b):
            angle_diff = abs(theta_b - theta_a)
            return np.cos(angle_diff)

        def get_label_position(vec, label, base_offset=0.15):
            start = vec.get_start()
            end = vec.get_end()
            direction = (end - start) / np.linalg.norm(end - start)
            # Calculate additional offset based on label width
            label_width = label.width
            # Add a little padding
            padding = 0.1
            total_offset = base_offset + label_width / 2 + padding
            return end + total_offset * direction

        def blend_colors(color1, color2, t):
            """Blend two Manim colors based on t in [0,1]."""
            rgb1 = np.array(color_to_rgb(color1))
            rgb2 = np.array(color_to_rgb(color2))
            blended = (1 - t) * rgb1 + t * rgb2
            return rgb_to_color(tuple(blended))


        # constants
        radius = 2
        center_shift = LEFT * 1.5

        angle_a = ValueTracker(0 * DEGREES)
        angle_b = ValueTracker(30 * DEGREES)

        # unit circle
        circle = Circle(radius=radius, color=YELLOW)
        x_axis = Line(LEFT*2.5, RIGHT*2.5, color=GREY).move_to(DOWN*0.01)
        y_axis = Line(DOWN*2.5, UP*2.5, color=GREY).move_to(RIGHT*0.01)
        circle_group = VGroup(circle, x_axis, y_axis).move_to(ORIGIN)

        # vectors
        reference_vector = always_redraw(lambda: Arrow(
            start=ORIGIN + center_shift,
            end=radius * np.array([
                np.cos(angle_a.get_value()),
                np.sin(angle_a.get_value()),
                0]) + center_shift,
            buff=0,
            color=YELLOW
        ))

        vector = always_redraw(lambda: Arrow(
            start=ORIGIN + center_shift,
            end=radius * np.array([
                np.cos(angle_b.get_value()),
                np.sin(angle_b.get_value()),
                0]) + center_shift,
            buff=0,
            color=YELLOW
        ))

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
                reference_vector, vector,
                radius=0.6,
                other_angle=use_other_angle,
                color=color,
                stroke_width=stroke_width
            )

        arc = always_redraw(get_arc)

        # angles and projections
        shadow_projection = always_redraw(lambda: DashedLine(
            start=radius * np.array([
                np.cos(angle_b.get_value()),
                np.sin(angle_b.get_value()),
                0]) + center_shift,
            end=[radius * np.cos(angle_b.get_value()), 0, 0] + center_shift,
            color=WHITE
        ))
        cos_brace = always_redraw(lambda: BraceBetweenPoints(
            ORIGIN + center_shift + DOWN * 0.02,
            [radius * np.cos(angle_b.get_value()), 0, 0] + center_shift + DOWN * 0.02,
            direction=DOWN,
            color=BLUE
        ))
        cos_brace_label = always_redraw(
            lambda: MathTex(f"{np.cos(angle_b.get_value()):.2f}", color=BLUE)\
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
                cos_bar.point_from_proportion((compute_cosine_similarity(angle_a.get_value(),angle_b.get_value()) + 1) / 2)
            )
        )
        cos_dist_dot.add_updater(
            lambda d: d.move_to(
                cos_bar.point_from_proportion((1 - compute_cosine_similarity(angle_a.get_value(),angle_b.get_value())) / 2)
            )
        )

        # formulas
        cos_label_lhs = always_redraw(
            lambda: MathTex(f"\\cos({int(np.degrees(angle_b.get_value() - angle_a.get_value()))}^\\circ) = ") \
                .scale(0.7) \
                .next_to(cos_bar, UP)
        )
        cos_label_rhs = always_redraw(
            lambda: MathTex(f"{compute_cosine_similarity(angle_a.get_value(),angle_b.get_value()):.2f}", color=BLUE) \
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
        cos_bar = baseline
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
        angles = [0.5, 90, 180, 60, 45, 120, 150]
        for deg in angles:
            theta = deg * DEGREES
            self.play(angle_b.animate.set_value(theta), run_time=1.5)
            self.wait()

        cos_dot.clear_updaters()
        cos_label_rhs.clear_updaters()
        cos_label_dist_rhs = always_redraw(
            lambda: MathTex(f"{1 - np.cos(angle_b.get_value()):.2f}", color=ORANGE) \
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
        angles= [0.5, 90, 180, 60, 45, 120, 150]
        for deg in angles:
            theta = deg * DEGREES
            self.play(angle_b.animate.set_value(theta), run_time=1.5)
            self.wait()
        self.wait(2)

        cos_bar_group = VGroup(
            one_minus,
            cos_label_dist_rhs,
            zero_label,
            plus_two_label,
            cos_dist_dot,
            cos_dist_text,
            cos_bar_text,
            cos_label_lhs,
            top_tick,
            bottom_tick,
            cos_bar
        )
        self.play(
            FadeOut(circle_group),
            FadeOut(shadow_projection),
            FadeOut(cos_brace),
            FadeOut(cos_brace_label),
            FadeOut(cos_bar_group),
        )
        self.play(self.camera.frame.animate.move_to(vector.get_start()).set(width=arc.width * 12))

        word_a = Text("king", font_size=32)
        word_b = Text("queen", font_size=32)
        word_a = word_a.move_to(get_label_position(reference_vector, word_a))
        word_b = word_b.move_to(get_label_position(vector, word_b))
        self.play(Write(word_a))
        self.play(Write(word_b))
        self.wait()
        transitions = [
            (PI / 8, -PI / 10, "dog", "cat"),
            (-PI / 6, PI / 5, "fish", "banana"),
            (PI / 4, PI / 3, "car", "warrior"),
            (-PI / 4, PI / 3, "basketball", "wanda"),
        ]

        for da, db, label_a, label_b in transitions:
            new_word_a = Text(label_a, font_size=32)
            new_word_b = Text(label_b, font_size=32)

            # Move to positions accounting for label width
            new_word_a.move_to(get_label_position(reference_vector, new_word_a))
            new_word_b.move_to(get_label_position(vector, new_word_b))

            # Updaters to keep labels following vectors with offset
            new_word_a.add_updater(lambda m: m.move_to(get_label_position(reference_vector, m)))
            new_word_b.add_updater(lambda m: m.move_to(get_label_position(vector, m)))

            self.play(
                Unwrite(word_a),
                Unwrite(word_b),
            )
            self.play(
                angle_a.animate.increment_value(da),
                angle_b.animate.increment_value(db),
                run_time=1,
                rate_func=smooth
            )
            self.play(
                Write(new_word_a),
                Write(new_word_b),
            )

            word_a = new_word_a
            word_b = new_word_b
            self.wait(0.5)


from manim import *
import string
import random
class MatrixZoomAndScan(MovingCameraScene):
    def construct(self):
        rows = 10
        cols = 100
        cell_w = 0.5
        cell_h = 0.4
        spacing = 0.05
        empty_tex_array = "".join(
            [
                r"\begin{array}{c}",
                *4 * [r"\quad \\"],
                r"\end{array}",
            ]
        )
        tex_left = "".join(
            [
                r"\left" + "[",
                empty_tex_array,
                r"\right.",
            ]
        )
        tex_right = "".join(
            [
                r"\left.",
                empty_tex_array,
                r"\right" + "]",
            ]
        )

        matrix_group = VGroup()
        source_numbers_matrix = []
        decimal_matrix = []

        for i in range(rows):
            row_trackers = []
            row_decimals = []
            for j in range(cols):
                tracker = ValueTracker(np.random.normal())
                decimal = DecimalNumber(tracker.get_value(), num_decimal_places=2, include_sign=True)
                decimal.scale(0.3)
                decimal.move_to(RIGHT * j * (cell_w + spacing) + DOWN * i * (cell_h + spacing))

                # Add updater to keep it in sync with the tracker
                decimal.add_updater(lambda d, t=tracker: d.set_value(t.get_value()))

                row_trackers.append(tracker)
                row_decimals.append(decimal)
            source_numbers_matrix.append(row_trackers)
            decimal_matrix.append(row_decimals)

        # Flatten into columns
        for j in range(cols):
            col_group = VGroup(*[decimal_matrix[i][j] for i in range(rows)])
            matrix_group.add(col_group)

        matrix_group.move_to(ORIGIN)
        l_bracket = MathTex(tex_left).next_to(matrix_group, LEFT)
        r_bracket = MathTex(tex_right).next_to(matrix_group, RIGHT)

        l_bracket.stretch_to_fit_height(matrix_group.height + 0.5)
        r_bracket.stretch_to_fit_height(matrix_group.height + 0.5)
        self.play(FadeIn(matrix_group), FadeIn(l_bracket), FadeIn(r_bracket))

        vocab_words = [
            'voids', 'enjoying', 'formula', 'suit', 'promptly', 'typical', 'richly', 'schedule', 'beacon', 'sharks',
            'hustle', 'dazzler', 'waving', 'become', 'sizzle', 'rocket', 'rulers', 'british', 'dividend', 'vocalize',
            'sheep', 'gaggle', 'reunion', 'coding', 'invite', 'servant', 'unheated', 'shaking', 'stratify', 'elvis',
            'insanely', 'shell', 'sighted', 'actions', 'cheery', 'volts', 'babbled', 'appeals', 'commoner', 'rumor',
            'asking', 'lurking', 'batch', 'entire', 'resolves', 'injury', 'glance', 'shrinks', 'fact', 'times',
            'flicks', 'tapping', 'initials', 'macbeth', 'shifts', 'lawns', 'anchored', 'hounds', 'pry', 'playoff',
            'compile', 'debated', 'gauche', 'infect', 'duffers', 'grandeur', 'arable', 'agonized', 'levity', 'activity',
            'melamine', 'shun', 'matunuck', 'whitcomb', 'disrupt', 'talker', 'files', 'junkies', 'voting', 'party', 'moll',
            'newlywed', 'daddy', 'treating', 'evelyn', 'resigns', 'climes', 'ignorant', 'korea', 'gris', 'nurture',
            'primeval', 'frontage', 'norborne', 'birthday', 'bemoan', 'hasher', 'lethargy', '...', 'prevent'
        ]
        label_texts = []

        for j, word in enumerate(vocab_words):
            label = Text(word, font_size=16).rotate(PI/3, axis=OUT)
            label.next_to(decimal_matrix[0][j], UP, buff=0.1).shift(UP*0.33)
            label_texts.append(label)

        label_group = VGroup(*label_texts)
        self.play(Write(label_group))
        self.wait()

        # Step 1: Zoomed out view
        self.play(self.camera.frame.animate.scale(6).move_to(matrix_group.get_center()), run_time=2)

        # Step 2: Zoom into first column
        first_col_center = matrix_group[0][rows//3].get_center()
        self.play(self.camera.frame.animate.scale(0.15).move_to(first_col_center), run_time=2)

        # Step 3: Animate values changing while scanning
        new_value_anims = []
        for i in range(rows):
            for j in range(cols):
                new_val = np.random.normal()
                new_value_anims.append(source_numbers_matrix[i][j].animate.set_value(new_val))

        last_col_center = matrix_group[-1][rows//3].get_center()

        self.play(
            AnimationGroup(
                self.camera.frame.animate.move_to(last_col_center),
                *new_value_anims,
                lag_ratio=0.0,
            ),
            run_time=13,
            rate_func=linear
        )

        self.wait()
        self.play(self.camera.frame.animate.scale(1/0.15).move_to(matrix_group.get_center()), run_time=2)
        self.wait()


np.random.seed(1)


from manim import *
import numpy as np

np.random.seed(1)

class CheckerboardDiskInSphere(ThreeDScene):
    @staticmethod
    def slerp(p1, p2, t):
        p1 = normalize(p1)
        p2 = normalize(p2)
        dot = np.clip(np.dot(p1, p2), -1.0, 1.0)
        theta = np.arccos(dot)
        if np.isclose(theta, 0):
            return p1
        return (
                np.sin((1 - t) * theta) / np.sin(theta) * p1 +
                np.sin(t * theta) / np.sin(theta) * p2
        )

    def create_spherical_arc(self, start, end, radius, thickness=0.01, steps=100, gradient=(YELLOW, ORANGE)):
        arc_group = VGroup()

        for i, t in enumerate(np.linspace(0, 1, steps - 1)):
            t_next = np.linspace(0, 1, steps)[i + 1]
            p1 = radius * self.slerp(start, end, t)
            p2 = radius * self.slerp(start, end, t_next)
            color = interpolate_color(gradient[0], gradient[1], i / (steps - 2))
            seg = Line3D(p1, p2, thickness=thickness, color=color)
            arc_group.add(seg)

        return arc_group

    def construct(self):
        def polar_to_point(r, theta):
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = 0  # XY plane
            return np.array([x, y, z])

        def get_tile_color(r, theta):
            ring_idx = int(r * 5)
            sector_idx = int(theta / TAU * 20)
            if (ring_idx + sector_idx) % 2 == 0:
                return ORANGE
            else:
                return RED

        # Create surface in polar coordinates mapped to XY
        def param_to_point(u, v):
            r = u * radius  # from 0 to radius
            theta = v * TAU  # from 0 to 2π
            return polar_to_point(r, theta)

        def edge_point(center, fraction=0.92):
            direction = normalize(center)
            return radius * fraction * direction

        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        axes = ThreeDAxes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            z_range=[-3, 3, 1],
            x_length=10,
            y_length=10,
            z_length=7
        )
        radius = 2
        sphere = Sphere(
            radius=radius,
            resolution=(40, 30),
            fill_opacity=0.55,
            checkerboard_colors=[BLUE_E],
            stroke_color=WHITE,
            stroke_width=0.1
        )
        disk = Surface(
            param_to_point,
            u_range=[0, 1],
            v_range=[0, 1],
            resolution=(10, 40),
        )
        for submob in disk.family_members_with_points():
            points = submob.get_points()
            colors = []
            for p in points:
                x, y = p[0], p[1]
                r = np.sqrt(x**2 + y**2) / radius
                theta = (np.arctan2(y, x) + TAU) % TAU
                colors.append(get_tile_color(r, theta))
            submob.set_color_by_rgba_array(np.array([c.to_rgba() for c in colors]))
        self.add(axes, sphere, disk)

        tile_a = 80
        tile_b = 102
        tile_c = 380
        self.move_camera(phi=60 * DEGREES, theta=30 * DEGREES, run_time=2)
        self.play(
            sphere[tile_a].animate.set_fill(YELLOW).set_opacity(1),
            run_time=1
        )
        self.wait()
        self.move_camera(phi=95 * DEGREES, theta=33 * DEGREES, run_time=2)
        self.play(
            sphere[tile_b].animate.set_fill(ORANGE).set_opacity(1),
            run_time=1
        )
        self.wait()
        self.move_camera(phi=60 * DEGREES, theta=105 * DEGREES, run_time=2)
        self.play(
            sphere[tile_c].animate.set_fill(GREEN_D).set_opacity(1),
            run_time=1
        )
        self.wait()
        self.move_camera(phi=75 * DEGREES, theta=45 * DEGREES, run_time=2)
        self.wait()


        # Get tile centers
        center_a = sphere[tile_a].get_center()
        center_b = sphere[tile_b].get_center()
        center_c = sphere[tile_c].get_center()

        # Pull the endpoints slightly inward toward the center of the sphere
        edge_a = edge_point(center_a)
        edge_b = edge_point(center_b)
        edge_c = edge_point(center_c)

        vec_1_stat = Arrow3D(start=ORIGIN, end=edge_a, color=BLUE_C, thickness=0.01)
        vec_1a = Arrow3D(start=ORIGIN, end=edge_a, color=BLUE_C, thickness=0.01)
        vec_1b = Arrow3D(start=ORIGIN, end=edge_a, color=BLUE_C, thickness=0.01)
        vec_2 = Arrow3D(start=ORIGIN, end=edge_b, color=BLUE_C, thickness=0.01)
        vec_3 = Arrow3D(start=ORIGIN, end=edge_c, color=BLUE_C, thickness=0.01)
        self.play(FadeIn(vec_1_stat), FadeIn(vec_1a), FadeIn(vec_1b))
        self.wait()

        arc_1 = self.create_spherical_arc(edge_a, edge_b, radius, gradient=[YELLOW,ORANGE], thickness=0.01)
        arc_2 = self.create_spherical_arc(edge_a, edge_c, radius, gradient=[YELLOW, GREEN], thickness=0.01)

        max_len = PI * radius
        slider_height = 3
        arc_bar = Line(ORIGIN, UP * slider_height, color=WHITE).shift(RIGHT * 5 + DOWN * 1.5).set_opacity(0)
        top_tick = Line(arc_bar.get_end() + LEFT * 0.15, arc_bar.get_end() + RIGHT * 0.15, color=WHITE).set_opacity(0)
        bottom_tick = Line(arc_bar.get_start() + LEFT * 0.15, arc_bar.get_start() + RIGHT * 0.15, color=WHITE).set_opacity(0)

        self.add_fixed_in_frame_mobjects(arc_bar, top_tick, bottom_tick)

        # === Dot that represents arc length magnitude
        dot_1 = Dot(radius=0.12, color=ORANGE).set_opacity(0)
        dot_2 = Dot(radius=0.12, color=GREEN).set_opacity(0)
        arc_1_len_tracker = ValueTracker(0)
        arc_2_len_tracker = ValueTracker(0)

        dot_1.add_updater(lambda d: d.move_to(
            arc_bar.point_from_proportion(
                np.clip(arc_1_len_tracker.get_value() / max_len, 0, 1)
            )
        ))
        dot_2.add_updater(lambda d: d.move_to(
            arc_bar.point_from_proportion(
                np.clip(arc_2_len_tracker.get_value() / max_len, 0, 1)
            )
        ))

        self.add_fixed_in_frame_mobjects(dot_1, dot_2)

        self.play(
            arc_bar.animate.set_opacity(1),
            top_tick.animate.set_opacity(1),
            bottom_tick.animate.set_opacity(1),
            dot_1.animate.set_opacity(1),
            dot_2.animate.set_opacity(1),
        )
        self.wait()
        self.move_camera(phi=85 * DEGREES, theta=30 * DEGREES, run_time=2)
        self.play(
            Create(arc_1),
            Transform(vec_1a, vec_2),
            arc_1_len_tracker.animate.set_value(max_len * 0.2),
            run_time=3
        )
        self.wait()
        self.move_camera(phi=80 * DEGREES, theta=80 * DEGREES, run_time=2)
        self.play(
            Create(arc_2),
            Transform(vec_1b, vec_3),
            arc_2_len_tracker.animate.set_value(max_len * 0.45),
            run_time=3
        )
        self.wait()
        self.move_camera(phi=75 * DEGREES, theta=45 * DEGREES, run_time=2)
        self.wait()
        self.play(
            FadeOut(vec_1a, vec_1b, vec_1_stat, vec_2, vec_3),
            FadeOut(arc_1, arc_2),
            FadeOut(dot_1, dot_2),
            FadeOut(arc_bar, top_tick, bottom_tick),
            sphere[tile_a].animate.set_fill(BLUE_E).set_opacity(0.55),
            sphere[tile_b].animate.set_fill(BLUE_E).set_opacity(0.55),
            sphere[tile_c].animate.set_fill(BLUE_E).set_opacity(0.55)
        )

        tiles = [24, 33, 35, 111, 118, 146, 199, 229, 238, 377, 428, 478, 518, 531, 751, 886, 919, 1070, 1126, 1191, 1198]
        self.play(*[sphere[tile].animate.set_fill(BLUE_C).set_opacity(1) for tile in tiles])
        self.move_camera(phi=75 * DEGREES, theta=15 * DEGREES, run_time=2)
        self.play(sphere[111].animate.set_fill(YELLOW).set_opacity(1))

        arcs = []
        primary_arc = None
        for tile in tiles:
            if tile == 111:
                continue
            center_a = sphere[111].get_center()
            center_b = sphere[tile].get_center()
            edge_a = edge_point(center_a)
            edge_b = edge_point(center_b)
            arc = self.create_spherical_arc(edge_a, edge_b, radius, gradient=[WHITE, WHITE], thickness=0.005)
            if tile == 1191:
                primary_arc = arc
            else:
                arcs.append(arc)
        self.play(*[Create(arc) for arc in arcs] + [Create(primary_arc)],run_time=2)
        self.wait(0.5)
        self.play(
            *[FadeOut(arc) for arc in arcs] + [primary_arc.animate.set_color(YELLOW).set_thickness(0.04)] +
            [sphere[1191].animate.set_fill(YELLOW).set_opacity(1)]
        )
        self.wait(0.5)
        self.play(
            FadeOut(primary_arc),
            sphere[1191].animate.set_fill(BLUE_C).set_opacity(1),
            sphere[111].animate.set_fill(BLUE_C).set_opacity(1)
        )
        self.wait(0.5)
        self.move_camera(phi=75 * DEGREES, theta=160 * DEGREES, run_time=2)
        self.play(sphere[531].animate.set_fill(YELLOW).set_opacity(1))
        arcs = []
        primary_arc = None
        for tile in tiles:
            if tile == 531:
                continue
            center_a = sphere[531].get_center()
            center_b = sphere[tile].get_center()
            edge_a = edge_point(center_a)
            edge_b = edge_point(center_b)
            arc = self.create_spherical_arc(edge_a, edge_b, radius, gradient=[WHITE, WHITE], thickness=0.005)
            if tile == 377:
                primary_arc = arc
            else:
                arcs.append(arc)
        self.play(*[Create(arc) for arc in arcs] + [Create(primary_arc)], run_time=2)
        self.wait(0.5)
        self.play(
            *[FadeOut(arc) for arc in arcs] + [primary_arc.animate.set_color(YELLOW).set_thickness(0.04)] +
             [sphere[377].animate.set_fill(YELLOW).set_opacity(1)]
        )
        self.wait(0.5)
        self.play(
            FadeOut(primary_arc),
            sphere[377].animate.set_fill(BLUE_C).set_opacity(1),
            sphere[531].animate.set_fill(BLUE_C).set_opacity(1)
        )
        self.wait()
